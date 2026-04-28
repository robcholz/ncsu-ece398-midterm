"""Evaluate the exported CMSIS-NN C model on the host validation split."""

from __future__ import annotations

import argparse
import ctypes
import json
import platform
import re
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from benchmark.host import apply_task, apply_train_normalization_with_stats, predict
from model.cnn import build_model
from model.dataset import WindowConfig, build_windows, subject_holdout_split
from model.metrics import classification_report


SOURCE_DIRS = (
    "ActivationFunctions",
    "BasicMathFunctions",
    "ConvolutionFunctions",
    "FullyConnectedFunctions",
    "NNSupportFunctions",
    "PoolingFunctions",
)


def main() -> int:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    split = load_split(args, checkpoint)
    c_lib = build_host_library(args)
    c_logits = run_c_model(c_lib, split.x_val, parse_input_scale(args.weights))
    c_pred = c_logits.argmax(axis=1)

    model = build_model(
        checkpoint["model"],
        in_channels=checkpoint["input_shape"][0],
        num_classes=len(checkpoint["class_names"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(split.x_val), torch.from_numpy(split.y_val)),
        batch_size=args.batch_size,
    )
    y_true, float_pred = predict(model, loader, torch.device("cpu"))

    class_names = tuple(checkpoint["class_names"])
    result = {
        "checkpoint": str(args.checkpoint),
        "weights": str(args.weights),
        "validation_windows": int(len(y_true)),
        "float_metrics": serialize_report(classification_report(y_true, float_pred, class_names)),
        "quantized_c_metrics": serialize_report(classification_report(y_true, c_pred, class_names)),
        "float_vs_quantized_c": {
            "prediction_agreement": float(np.mean(float_pred == c_pred)) if len(c_pred) else 0.0,
            "changed_predictions": int(np.count_nonzero(float_pred != c_pred)),
        },
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=Path("model/cmsis/imu_model_weights.h"))
    parser.add_argument("--data-root", type=Path, default=Path("dataset/Multimodal Cough Dataset"))
    parser.add_argument("--build-dir", type=Path, default=Path("target/cmsis-host"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--compiler", default="clang")
    return parser.parse_args()


def load_split(args: argparse.Namespace, checkpoint: dict):
    config = checkpoint["config"]
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
        seed=398,
    )
    x, y, metadata = build_windows(args.data_root, window_config, max_windows=args.max_windows)
    split = subject_holdout_split(x, y, metadata)
    split = apply_task(split, config["task"])
    if config["normalization"] == "train":
        split, _stats = apply_train_normalization_with_stats(split)
    return split


def build_host_library(args: argparse.Namespace) -> Path:
    args.build_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    output = args.build_dir / f"libimu_model_host{suffix}"

    cmsis_nn = Path("third_party/CMSIS-NN")
    cmsis_core = Path("third_party/CMSIS_6/CMSIS/Core/Include")
    sources = [Path("model/cmsis/imu_model.c")]
    for source_dir in SOURCE_DIRS:
        sources.extend(sorted((cmsis_nn / "Source" / source_dir).glob("*.c")))

    cmd = [
        args.compiler,
        "-shared",
        "-fPIC",
        "-O3",
        "-DCMSIS_NN_USE_SINGLE_ROUNDING",
        "-I",
        "model/cmsis",
        "-I",
        str(cmsis_nn / "Include"),
        "-I",
        str(cmsis_core),
        "-o",
        str(output),
        *[str(source) for source in sources],
    ]
    subprocess.run(cmd, check=True)
    return output


def parse_input_scale(weights: Path) -> float:
    text = weights.read_text(encoding="utf-8")
    match = re.search(r"#define\s+IMU_MODEL_INPUT_SCALE\s+([0-9.eE+-]+)f", text)
    if not match:
        raise ValueError(f"Could not find IMU_MODEL_INPUT_SCALE in {weights}")
    return float(match.group(1))


def run_c_model(lib_path: Path, x_val: np.ndarray, input_scale: float) -> np.ndarray:
    lib = ctypes.CDLL(str(lib_path.resolve()))
    lib.imu_model_run.argtypes = [
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_int8),
    ]
    lib.imu_model_run.restype = ctypes.c_int

    logits = np.empty((len(x_val), 8), dtype=np.int8)
    for idx, window in enumerate(x_val):
        quantized = quantize_window(window, input_scale)
        output = np.zeros(8, dtype=np.int8)
        status = lib.imu_model_run(
            quantized.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        )
        if status != 0:
            raise RuntimeError(f"imu_model_run failed for window {idx}: status={status}")
        logits[idx] = output
    return logits


def quantize_window(window: np.ndarray, input_scale: float) -> np.ndarray:
    flattened = window.transpose(1, 0).reshape(-1)
    return np.round(flattened / input_scale).clip(-128, 127).astype(np.int8)


def serialize_report(report: dict) -> dict:
    return {
        **report,
        "confusion_matrix": report["confusion_matrix"].astype(int).tolist(),
    }


if __name__ == "__main__":
    raise SystemExit(main())
