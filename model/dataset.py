"""Dataset windowing for the multimodal cough accelerometer data."""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

BACKGROUND_LABEL = "background"
EVENT_LABELS = (
    "Cough",
    "Speech",
    "Sneeze",
    "Deep breath",
    "Groan",
    "Laugh",
    "Speech (far)",
)
LABELS = (BACKGROUND_LABEL, *EVENT_LABELS)
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(LABELS)}

_OTHER_SOUND = "Other Sound"
_ACCEL_COLUMNS = ("x-axis (g)", "y-axis (g)", "z-axis (g)")


@dataclass(frozen=True)
class WindowConfig:
    sample_rate_hz: int = 100
    window_seconds: float = 2.0
    stride_seconds: float = 0.5
    label_overlap_threshold: float = 0.3
    include_magnitude: bool = False
    normalize: bool = False
    max_background_ratio: float | None = 3.0
    sampling_strategy: str = "sliding"
    event_windows_per_event: int = 3
    event_jitter_seconds: float = 0.25
    background_windows_per_event: float = 1.0
    background_exclusion_seconds: float = 0.25
    seed: int = 398

    @property
    def samples_per_window(self) -> int:
        return int(round(self.sample_rate_hz * self.window_seconds))


@dataclass(frozen=True)
class Event:
    start: float
    end: float
    label: str


@dataclass(frozen=True)
class Recording:
    subject: str
    trial: str
    trial_dir: Path
    accelerometer_csv: Path
    sync_data_label_start: float
    sync_imu_start: float
    events: tuple[Event, ...]


@dataclass(frozen=True)
class BenchmarkSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    class_names: tuple[str, ...]
    train_subjects: tuple[str, ...]
    val_subjects: tuple[str, ...]


def discover_recordings(data_root: str | Path) -> list[Recording]:
    """Find annotated accelerometer recordings under the Dryad dataset root."""

    root = Path(data_root)
    annotations_path = root / "DataAnnotation.json"
    with annotations_path.open("r", encoding="utf-8") as f:
        annotations = json.load(f)

    by_key: dict[tuple[str, str], list[Event]] = {}
    for item in annotations:
        filename = item["original_filename"]
        subject, trial = _parse_annotation_filename(filename)
        sync = _read_sync(root / subject / "sync_time.txt")[trial]
        for segment in item.get("segmentations", []):
            label = _segment_label(segment)
            if label is None or label == _OTHER_SOUND:
                continue
            start = sync["imu_start"] + (
                float(segment["start_time"]) - sync["data_label_start"]
            )
            end = sync["imu_start"] + (
                float(segment["end_time"]) - sync["data_label_start"]
            )
            by_key.setdefault((subject, trial), []).append(Event(start, end, label))

    recordings: list[Recording] = []
    for (subject, trial), events in sorted(by_key.items()):
        sync = _read_sync(root / subject / "sync_time.txt")[trial]
        trial_dir = root / subject / _trial_dir_name(trial)
        accelerometer_csv = trial_dir / "Accelerometer.csv"
        if not accelerometer_csv.exists():
            continue
        recordings.append(
            Recording(
                subject=subject,
                trial=trial,
                trial_dir=trial_dir,
                accelerometer_csv=accelerometer_csv,
                sync_data_label_start=sync["data_label_start"],
                sync_imu_start=sync["imu_start"],
                events=tuple(sorted(events, key=lambda event: event.start)),
            )
        )
    return recordings


def build_windows(
    data_root: str | Path,
    config: WindowConfig = WindowConfig(),
    max_windows: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    """Build fixed-length windows and integer labels.

    Returns ``x`` as ``[num_windows, channels, samples]`` for direct Conv1D use.
    Metadata currently includes subject and trial strings for split strategies.
    """

    rng = random.Random(config.seed)
    windows: list[np.ndarray] = []
    labels: list[int] = []
    metadata: list[dict[str, str]] = []

    if config.sampling_strategy not in {"sliding", "event-centered"}:
        raise ValueError(f"Unknown sampling strategy: {config.sampling_strategy}")

    background_indices: list[int] = []
    event_count = 0
    for recording in discover_recordings(data_root):
        times, values = _read_accelerometer(recording.accelerometer_csv)
        if len(times) < config.samples_per_window:
            continue

        start = max(float(times[0]), recording.sync_imu_start)
        end = float(times[-1]) - config.window_seconds
        if end <= start:
            continue

        if config.sampling_strategy == "sliding":
            added_events = _add_sliding_windows(
                windows,
                labels,
                metadata,
                background_indices,
                recording,
                times,
                values,
                start,
                end,
                config,
            )
        else:
            added_events = _add_event_centered_windows(
                windows,
                labels,
                metadata,
                background_indices,
                rng,
                recording,
                times,
                values,
                start,
                end,
                config,
            )
        event_count += added_events

    keep = list(range(len(labels)))
    if config.max_background_ratio is not None and event_count > 0:
        max_background = int(math.ceil(event_count * config.max_background_ratio))
        if len(background_indices) > max_background:
            selected_background = set(rng.sample(background_indices, max_background))
            keep = [
                idx
                for idx, label in enumerate(labels)
                if label != 0 or idx in selected_background
            ]

    if max_windows is not None and len(keep) > max_windows:
        keep = sorted(rng.sample(keep, max_windows))

    if not keep:
        channels = 4 if config.include_magnitude else 3
        return (
            np.empty((0, channels, config.samples_per_window), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            [],
        )

    x = np.stack([windows[idx] for idx in keep]).astype(np.float32, copy=False)
    y = np.asarray([labels[idx] for idx in keep], dtype=np.int64)
    kept_metadata = [metadata[idx] for idx in keep]
    return x, y, kept_metadata


def _add_sliding_windows(
    windows: list[np.ndarray],
    labels: list[int],
    metadata: list[dict[str, str]],
    background_indices: list[int],
    recording: Recording,
    times: np.ndarray,
    values: np.ndarray,
    start: float,
    end: float,
    config: WindowConfig,
) -> int:
    event_count = 0
    cursor = start
    while cursor <= end:
        window = _window_at(times, values, cursor, config)
        label = _label_window(
            cursor,
            cursor + config.window_seconds,
            recording.events,
            config,
        )
        if label == 0:
            background_indices.append(len(labels))
        else:
            event_count += 1
        windows.append(window)
        labels.append(label)
        metadata.append(
            {"subject": recording.subject, "trial": recording.trial, "source": "sliding"}
        )
        cursor += config.stride_seconds
    return event_count


def _add_event_centered_windows(
    windows: list[np.ndarray],
    labels: list[int],
    metadata: list[dict[str, str]],
    background_indices: list[int],
    rng: random.Random,
    recording: Recording,
    times: np.ndarray,
    values: np.ndarray,
    start: float,
    end: float,
    config: WindowConfig,
) -> int:
    event_count = 0
    for event in recording.events:
        center = (event.start + event.end) / 2
        for _ in range(max(config.event_windows_per_event, 1)):
            jitter = rng.uniform(-config.event_jitter_seconds, config.event_jitter_seconds)
            window_start = _clamp_window_start(
                center - config.window_seconds / 2 + jitter,
                start,
                end,
            )
            windows.append(_window_at(times, values, window_start, config))
            labels.append(LABEL_TO_INDEX[event.label])
            metadata.append(
                {
                    "subject": recording.subject,
                    "trial": recording.trial,
                    "source": "event",
                }
            )
            event_count += 1

    background_target = int(
        round(len(recording.events) * max(config.background_windows_per_event, 0.0))
    )
    attempts = 0
    added_background = 0
    while added_background < background_target and attempts < max(background_target * 30, 30):
        attempts += 1
        window_start = rng.uniform(start, end)
        window_end = window_start + config.window_seconds
        if _too_close_to_event(
            window_start,
            window_end,
            recording.events,
            config.background_exclusion_seconds,
        ):
            continue
        if _label_window(window_start, window_end, recording.events, config) != 0:
            continue
        windows.append(_window_at(times, values, window_start, config))
        labels.append(LABEL_TO_INDEX[BACKGROUND_LABEL])
        background_indices.append(len(labels) - 1)
        metadata.append(
            {
                "subject": recording.subject,
                "trial": recording.trial,
                "source": "background",
            }
        )
        added_background += 1
    return event_count


def _clamp_window_start(window_start: float, start: float, end: float) -> float:
    return min(max(window_start, start), end)


def _too_close_to_event(
    start: float,
    end: float,
    events: tuple[Event, ...],
    margin: float,
) -> bool:
    for event in events:
        if max(start, event.start - margin) < min(end, event.end + margin):
            return True
    return False


def subject_holdout_split(
    x: np.ndarray,
    y: np.ndarray,
    metadata: list[dict[str, str]],
    val_subjects: Iterable[str] | None = None,
) -> BenchmarkSplit:
    """Split windows by subject, defaulting to the last 20% of subjects for val."""

    subjects = tuple(sorted({item["subject"] for item in metadata}))
    if not subjects:
        raise ValueError("No subjects found in window metadata.")

    if val_subjects is None:
        val_count = max(1, math.ceil(len(subjects) * 0.2))
        val_subject_tuple = subjects[-val_count:]
    else:
        val_subject_tuple = tuple(sorted(val_subjects))
    val_subject_set = set(val_subject_tuple)
    train_subject_tuple = tuple(subject for subject in subjects if subject not in val_subject_set)

    val_mask = np.asarray([item["subject"] in val_subject_set for item in metadata])
    train_mask = ~val_mask
    if not train_mask.any() or not val_mask.any():
        raise ValueError("Subject split produced an empty train or validation set.")

    return BenchmarkSplit(
        x_train=x[train_mask],
        y_train=y[train_mask],
        x_val=x[val_mask],
        y_val=y[val_mask],
        class_names=LABELS,
        train_subjects=train_subject_tuple,
        val_subjects=val_subject_tuple,
    )


def _read_accelerometer(path: Path) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    values: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                times.append(float(row["elapsed (s)"]))
                values.append(tuple(float(row[column]) for column in _ACCEL_COLUMNS))
            except (KeyError, TypeError, ValueError):
                continue
    return np.asarray(times, dtype=np.float64), np.asarray(values, dtype=np.float32)


def _window_at(
    times: np.ndarray,
    values: np.ndarray,
    start: float,
    config: WindowConfig,
) -> np.ndarray:
    target_times = start + np.arange(config.samples_per_window) / config.sample_rate_hz
    channels = [
        np.interp(target_times, times, values[:, axis]).astype(np.float32)
        for axis in range(3)
    ]
    window = np.stack(channels, axis=0)
    if config.include_magnitude:
        magnitude = np.sqrt(np.sum(window * window, axis=0, keepdims=True))
        window = np.concatenate([window, magnitude], axis=0)
    if config.normalize:
        mean = window.mean(axis=1, keepdims=True)
        std = window.std(axis=1, keepdims=True)
        window = (window - mean) / np.maximum(std, 1e-6)
    return window


def _label_window(
    start: float,
    end: float,
    events: tuple[Event, ...],
    config: WindowConfig,
) -> int:
    best_label = BACKGROUND_LABEL
    best_overlap = 0.0
    for event in events:
        overlap = max(0.0, min(end, event.end) - max(start, event.start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = event.label

    if best_overlap / config.window_seconds < config.label_overlap_threshold:
        return LABEL_TO_INDEX[BACKGROUND_LABEL]
    return LABEL_TO_INDEX[best_label]


def _segment_label(segment: dict) -> str | None:
    annotations = segment.get("annotations", {})
    if not annotations:
        return None
    label = next(iter(annotations.keys()))
    return "Deep breath" if label == "Deep Breath" else label


def _parse_annotation_filename(filename: str) -> tuple[str, str]:
    normalized = filename.lower().replace("-", "_")
    subject = normalized[:3]
    if "no_talking" in normalized or "notalking" in normalized:
        return subject, "trial1"
    if "nonverbal" in normalized:
        return subject, "trial3"
    if "talking" in normalized:
        return subject, "trial2"
    raise ValueError(f"Cannot infer trial from annotation filename: {filename}")


def _trial_dir_name(trial: str) -> str:
    return {
        "trial1": "Trial_1_No_Talking",
        "trial2": "Trial_2_Talking",
        "trial3": "Trial_3_Nonverbal",
    }[trial]


def _read_sync(path: Path) -> dict[str, dict[str, float]]:
    sync: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6 or parts[0] != "trial":
                continue
            trial = f"trial{parts[1]}"
            sync[trial] = {
                "data_label_start": float(parts[3]),
                "imu_start": float(parts[5]),
            }
    return sync
