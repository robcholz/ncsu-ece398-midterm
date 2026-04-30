#!/usr/bin/env python3
"""Visualize the cough IMU dataset and its firmware input format."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))

import matplotlib.pyplot as plt
import numpy as np

from model.dataset import LABELS, WindowConfig, build_windows, discover_recordings

ACCEL_COLUMNS = ("x-axis (g)", "y-axis (g)", "z-axis (g)")
CHANNEL_NAMES_3 = ("ax_g", "ay_g", "az_g")
CHANNEL_NAMES_4 = ("ax_g", "ay_g", "az_g", "magnitude_g")
FEATURE_NAMES = ("mean", "std", "min", "max", "range", "rms", "abs_mean", "delta")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = WindowConfig(
        sample_rate_hz=args.sample_rate_hz,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride,
        label_overlap_threshold=args.label_overlap_threshold,
        include_magnitude=not args.no_magnitude,
        normalize=False,
        max_background_ratio=args.max_background_ratio,
        sampling_strategy=args.sampling_strategy,
        event_windows_per_event=args.event_windows_per_event,
        event_jitter_seconds=args.event_jitter_seconds,
        background_windows_per_event=args.background_windows_per_event,
        background_exclusion_seconds=args.background_exclusion_seconds,
        seed=args.seed,
    )
    windows, labels, metadata = build_windows(args.data_root, config, args.max_windows)
    if len(labels) == 0:
        raise SystemExit(f"No windows built from {args.data_root}")

    channel_names = CHANNEL_NAMES_4 if config.include_magnitude else CHANNEL_NAMES_3
    features, feature_columns = window_features(windows, channel_names)
    constants = load_firmware_constants(args.model_header, args.weights_header)
    preview_index = choose_preview_index(labels, args.preview_label)

    write_summary(args.output_dir / "summary.md", windows, labels, metadata, config, constants)
    write_feature_csv(
        args.output_dir / "window_features.csv",
        features,
        feature_columns,
        labels,
        metadata,
    )
    write_feature_summary_csv(
        args.output_dir / "feature_summary_by_class.csv",
        features,
        feature_columns,
        labels,
    )
    write_firmware_preview_csv(
        args.output_dir / "firmware_input_preview.csv",
        windows[preview_index],
        channel_names,
        constants,
    )

    plot_class_distribution(args.output_dir / "class_distribution.png", labels)
    plot_feature_pca(args.output_dir / "feature_pca.png", features, labels, args.max_plot_points)
    plot_feature_bars(
        args.output_dir / "feature_summary.png",
        features,
        feature_columns,
        labels,
        channel_names,
    )
    plot_sample_windows(
        args.output_dir / "sample_windows_by_class.png",
        windows,
        labels,
        channel_names,
    )
    plot_recording_timeline(
        args.output_dir / "recording_timeline.png",
        args.data_root,
        args.subject,
        args.trial,
        args.timeline_seconds,
    )

    print(f"wrote dataset visualization to {args.output_dir}")
    print("firmware input order: input[sample * CHANNELS + channel]")
    print(f"window tensor shape: {list(windows.shape)} = [windows, channels, samples]")
    print(f"channels: {', '.join(channel_names)}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("dataset/Multimodal Cough Dataset"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark/dataset_visualization"),
    )
    parser.add_argument("--max-windows", type=int, default=20000)
    parser.add_argument("--max-plot-points", type=int, default=5000)
    parser.add_argument("--sample-rate-hz", type=int, default=100)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--stride", type=float, default=0.5)
    parser.add_argument("--label-overlap-threshold", type=float, default=0.15)
    parser.add_argument("--no-magnitude", action="store_true")
    parser.add_argument("--max-background-ratio", type=float, default=3.0)
    parser.add_argument(
        "--sampling-strategy",
        choices=("event-centered", "sliding"),
        default="event-centered",
    )
    parser.add_argument("--event-windows-per-event", type=int, default=3)
    parser.add_argument("--event-jitter-seconds", type=float, default=0.25)
    parser.add_argument("--background-windows-per-event", type=float, default=1.0)
    parser.add_argument("--background-exclusion-seconds", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=398)
    parser.add_argument("--subject", help="Subject id for timeline plot, e.g. 005")
    parser.add_argument("--trial", help="Trial id for timeline plot, e.g. trial1")
    parser.add_argument("--timeline-seconds", type=float, default=20.0)
    parser.add_argument("--preview-label", default="Cough")
    parser.add_argument("--model-header", type=Path, default=Path("model/cmsis/imu_model.h"))
    parser.add_argument(
        "--weights-header",
        type=Path,
        default=Path("model/cmsis/imu_model_weights.h"),
    )
    return parser.parse_args()


def read_accelerometer(path: Path) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    values: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            try:
                times.append(float(row["elapsed (s)"]))
                values.append(tuple(float(row[column]) for column in ACCEL_COLUMNS))
            except (KeyError, TypeError, ValueError):
                continue
    return np.asarray(times, dtype=np.float64), np.asarray(values, dtype=np.float32)


def window_features(
    windows: np.ndarray,
    channel_names: tuple[str, ...],
) -> tuple[np.ndarray, list[str]]:
    columns: list[str] = []
    chunks: list[np.ndarray] = []
    for channel_index, channel_name in enumerate(channel_names):
        x = windows[:, channel_index, :]
        minimum = x.min(axis=1)
        maximum = x.max(axis=1)
        values = [
            x.mean(axis=1),
            x.std(axis=1),
            minimum,
            maximum,
            maximum - minimum,
            np.sqrt((x * x).mean(axis=1) + 1e-6),
            np.abs(x).mean(axis=1),
            x[:, -1] - x[:, 0],
        ]
        chunks.extend(values)
        columns.extend(f"{channel_name}_{feature}" for feature in FEATURE_NAMES)
    return np.stack(chunks, axis=1), columns


def load_firmware_constants(model_header: Path, weights_header: Path) -> dict:
    constants = {
        "input_len": None,
        "channels": None,
        "input_scale": None,
        "norm_mean": None,
        "norm_std": None,
    }
    if model_header.exists():
        text = model_header.read_text(encoding="utf-8")
        constants["input_len"] = parse_define_int(text, "IMU_MODEL_INPUT_LEN")
        constants["channels"] = parse_define_int(text, "IMU_MODEL_CHANNELS")
    if weights_header.exists():
        text = weights_header.read_text(encoding="utf-8")
        constants["input_scale"] = parse_define_float(text, "IMU_MODEL_INPUT_SCALE")
        constants["norm_mean"] = parse_float_array(text, "IMU_MODEL_NORM_MEAN")
        constants["norm_std"] = parse_float_array(text, "IMU_MODEL_NORM_STD")
    return constants


def parse_define_int(text: str, name: str) -> int | None:
    match = re.search(rf"#define\s+{name}\s+(\d+)", text)
    return int(match.group(1)) if match else None


def parse_define_float(text: str, name: str) -> float | None:
    match = re.search(rf"#define\s+{name}\s+([-+0-9.eE]+)f?", text)
    return float(match.group(1)) if match else None


def parse_float_array(text: str, name: str) -> list[float] | None:
    match = re.search(rf"{name}\[[^\]]+\]\s*=\s*\{{([^}}]+)\}}", text, re.MULTILINE)
    if not match:
        return None
    return [float(item.strip().rstrip("f")) for item in match.group(1).split(",")]


def choose_preview_index(labels: np.ndarray, preview_label: str) -> int:
    target = LABELS.index(preview_label) if preview_label in LABELS else None
    if target is not None:
        matches = np.flatnonzero(labels == target)
        if len(matches) > 0:
            return int(matches[0])
    event_matches = np.flatnonzero(labels != 0)
    if len(event_matches) > 0:
        return int(event_matches[0])
    return 0


def write_summary(
    path: Path,
    windows: np.ndarray,
    labels: np.ndarray,
    metadata: list[dict[str, str]],
    config: WindowConfig,
    constants: dict,
) -> None:
    counts = np.bincount(labels, minlength=len(LABELS))
    subjects = sorted({item["subject"] for item in metadata})
    trials = sorted({item["trial"] for item in metadata})
    lines = [
        "# Dataset Visualization Summary",
        "",
        "## Window Format",
        "",
        f"- Tensor shape: `{list(windows.shape)}` as `[windows, channels, samples]`.",
        f"- Sample rate: `{config.sample_rate_hz}` Hz.",
        f"- Window length: `{config.samples_per_window}` samples / `{config.window_seconds}` seconds.",
        f"- Channels: `{', '.join(CHANNEL_NAMES_4 if config.include_magnitude else CHANNEL_NAMES_3)}`.",
        "- Dataset CSV units: accelerometer columns are in `g`.",
        "- Firmware live path accepts m/s^2 in `push_sample_m_s2`, converts to `g`, computes magnitude, then quantizes.",
        "- Firmware model path expects time-major int8 layout: `input[sample * CHANNELS + channel]`.",
        "",
        "## Firmware Constants",
        "",
        f"- `IMU_MODEL_INPUT_LEN`: `{constants.get('input_len')}`.",
        f"- `IMU_MODEL_CHANNELS`: `{constants.get('channels')}`.",
        f"- `IMU_MODEL_INPUT_SCALE`: `{constants.get('input_scale')}`.",
        f"- `IMU_MODEL_NORM_MEAN`: `{constants.get('norm_mean')}`.",
        f"- `IMU_MODEL_NORM_STD`: `{constants.get('norm_std')}`.",
        "",
        "## Dataset Counts",
        "",
        f"- Windows: `{len(labels)}`.",
        f"- Subjects: `{', '.join(subjects)}`.",
        f"- Trials: `{', '.join(trials)}`.",
        "",
    ]
    for idx, label in enumerate(LABELS):
        lines.append(f"- `{label}`: `{int(counts[idx])}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_feature_csv(
    path: Path,
    features: np.ndarray,
    columns: list[str],
    labels: np.ndarray,
    metadata: list[dict[str, str]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["window", "label", "subject", "trial", "source", *columns])
        for index, row in enumerate(features):
            item = metadata[index]
            writer.writerow(
                [
                    index,
                    LABELS[int(labels[index])],
                    item.get("subject", ""),
                    item.get("trial", ""),
                    item.get("source", ""),
                    *[f"{value:.8g}" for value in row],
                ]
            )


def write_feature_summary_csv(
    path: Path,
    features: np.ndarray,
    columns: list[str],
    labels: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["label", "feature", "mean", "std", "median"])
        for label_index, label in enumerate(LABELS):
            class_features = features[labels == label_index]
            if len(class_features) == 0:
                continue
            for col_index, column in enumerate(columns):
                values = class_features[:, col_index]
                writer.writerow(
                    [
                        label,
                        column,
                        f"{values.mean():.8g}",
                        f"{values.std():.8g}",
                        f"{np.median(values):.8g}",
                    ]
                )


def write_firmware_preview_csv(
    path: Path,
    window: np.ndarray,
    channel_names: tuple[str, ...],
    constants: dict,
) -> None:
    norm_mean = constants.get("norm_mean")
    norm_std = constants.get("norm_std")
    input_scale = constants.get("input_scale")
    can_quantize = (
        norm_mean is not None
        and norm_std is not None
        and input_scale is not None
        and len(norm_mean) >= len(channel_names)
        and len(norm_std) >= len(channel_names)
    )
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "sample",
                "channel",
                "flat_index",
                "value_g",
                "normalized",
                "quantized_i8",
            ]
        )
        for sample in range(window.shape[1]):
            for channel, channel_name in enumerate(channel_names):
                value = float(window[channel, sample])
                normalized = ""
                quantized = ""
                if can_quantize:
                    normalized_value = (value - norm_mean[channel]) / norm_std[channel]
                    quantized_value = round_to_i8(normalized_value / input_scale)
                    normalized = f"{normalized_value:.8g}"
                    quantized = str(quantized_value)
                writer.writerow(
                    [
                        sample,
                        channel_name,
                        sample * len(channel_names) + channel,
                        f"{value:.8g}",
                        normalized,
                        quantized,
                    ]
                )


def round_to_i8(value: float) -> int:
    rounded = math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)
    return max(-128, min(127, rounded))


def plot_class_distribution(path: Path, labels: np.ndarray) -> None:
    counts = np.bincount(labels, minlength=len(LABELS))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(LABELS, counts, color="#4c78a8")
    ax.set_ylabel("windows")
    ax.set_title("Window Class Distribution")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_feature_pca(
    path: Path,
    features: np.ndarray,
    labels: np.ndarray,
    max_points: int,
) -> None:
    rng = np.random.default_rng(398)
    indices = np.arange(len(labels))
    if len(indices) > max_points:
        indices = np.sort(rng.choice(indices, size=max_points, replace=False))
    x = features[indices]
    y = labels[indices]
    x = (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 1e-6)
    _, _, vh = np.linalg.svd(x, full_matrices=False)
    projected = x @ vh[:2].T

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    for label_index, label in enumerate(LABELS):
        mask = y == label_index
        if mask.any():
            ax.scatter(
                projected[mask, 0],
                projected[mask, 1],
                s=10,
                alpha=0.65,
                color=cmap(label_index),
                label=label,
            )
    ax.set_title("Window Feature PCA")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_feature_bars(
    path: Path,
    features: np.ndarray,
    columns: list[str],
    labels: np.ndarray,
    channel_names: tuple[str, ...],
) -> None:
    wanted = [f"{channel_names[-1]}_rms", f"{channel_names[-1]}_std", f"{channel_names[-1]}_range"]
    column_indices = [columns.index(name) for name in wanted if name in columns]
    fig, axes = plt.subplots(len(column_indices), 1, figsize=(10, 3.2 * len(column_indices)))
    if len(column_indices) == 1:
        axes = [axes]
    for ax, col_index in zip(axes, column_indices, strict=False):
        means = []
        for label_index in range(len(LABELS)):
            values = features[labels == label_index, col_index]
            means.append(values.mean() if len(values) else 0.0)
        ax.bar(LABELS, means, color="#59a14f")
        ax.set_title(columns[col_index])
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_sample_windows(
    path: Path,
    windows: np.ndarray,
    labels: np.ndarray,
    channel_names: tuple[str, ...],
) -> None:
    rows = len(LABELS)
    fig, axes = plt.subplots(rows, 1, figsize=(10, 1.9 * rows), sharex=True)
    time_axis = np.arange(windows.shape[2]) / 100.0
    for label_index, ax in enumerate(axes):
        matches = np.flatnonzero(labels == label_index)
        if len(matches) == 0:
            ax.set_visible(False)
            continue
        window = windows[int(matches[0])]
        for channel, channel_name in enumerate(channel_names):
            ax.plot(time_axis, window[channel], linewidth=1.0, label=channel_name)
        ax.set_ylabel(LABELS[label_index], rotation=0, ha="right", va="center")
        ax.grid(alpha=0.2)
    axes[0].legend(ncol=len(channel_names), fontsize=8)
    axes[-1].set_xlabel("seconds")
    fig.suptitle("First Window Found Per Class")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_recording_timeline(
    path: Path,
    data_root: Path,
    subject: str | None,
    trial: str | None,
    timeline_seconds: float,
) -> None:
    recordings = discover_recordings(data_root)
    if subject:
        recordings = [recording for recording in recordings if recording.subject == subject]
    if trial:
        recordings = [recording for recording in recordings if recording.trial == trial]
    if not recordings:
        return

    recording = recordings[0]
    times, values = read_accelerometer(recording.accelerometer_csv)
    magnitude = np.sqrt(np.sum(values * values, axis=1))
    center = recording.events[0].start if recording.events else float(times[0])
    start = max(float(times[0]), center - timeline_seconds / 2)
    end = min(float(times[-1]), start + timeline_seconds)
    mask = (times >= start) & (times <= end)

    fig, ax = plt.subplots(figsize=(12, 5))
    relative_time = times[mask] - start
    for axis, name in enumerate(CHANNEL_NAMES_3):
        ax.plot(relative_time, values[mask, axis], linewidth=0.9, label=name)
    ax.plot(relative_time, magnitude[mask], linewidth=1.1, label="magnitude_g")

    ylim = ax.get_ylim()
    for event in recording.events:
        overlap_start = max(event.start, start)
        overlap_end = min(event.end, end)
        if overlap_start >= overlap_end:
            continue
        ax.axvspan(overlap_start - start, overlap_end - start, alpha=0.18)
        ax.text(
            overlap_start - start,
            ylim[1],
            event.label,
            fontsize=8,
            rotation=90,
            va="top",
        )
    ax.set_title(f"Recording Timeline: subject {recording.subject} {recording.trial}")
    ax.set_xlabel("seconds from plotted start")
    ax.set_ylabel("g")
    ax.grid(alpha=0.2)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
