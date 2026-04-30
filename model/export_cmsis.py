"""Export the trained small CNN to CMSIS-NN C weights.

This exporter targets the deployment wrapper in ``model/cmsis/imu_model.c``:

    input [1, 1, 100, 4]
    conv k=5 + relu + maxpool
    conv k=5 + relu + maxpool
    conv k=3 + relu + global avgpool
    fully-connected 64->8

The generated header uses symmetric int8 quantization and per-output-channel
weight quantization for convolution layers.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from benchmark.host import apply_task, apply_train_normalization
from model.cnn import build_model
from model.dataset import LABELS, WindowConfig, build_windows, subject_holdout_split


def main() -> int:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["model"] != "small":
        raise SystemExit("CMSIS exporter currently targets the small deployment CNN.")

    model = build_model(
        checkpoint["model"],
        in_channels=checkpoint["input_shape"][0],
        num_classes=len(checkpoint["class_names"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    if tuple(checkpoint["input_shape"]) != (4, 100):
        raise SystemExit(
            f"Expected checkpoint input_shape [4, 100], got {checkpoint['input_shape']}"
        )
    if tuple(checkpoint["class_names"]) != LABELS:
        raise SystemExit("CMSIS wrapper expects the 8-class multiclass label order.")

    calibration = load_calibration_windows(args, checkpoint)
    scales = calibrate_scales(model, calibration, args.activation_percentile)
    layers = extract_small_layers(model, scales)
    architecture = extract_deployment_architecture(model, checkpoint)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_header(checkpoint, architecture, scales, layers),
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("model/cmsis/imu_model_weights.h")
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("dataset/Multimodal Cough Dataset"),
        help="Dataset root used for calibration windows.",
    )
    parser.add_argument("--max-calibration-windows", type=int, default=2048)
    parser.add_argument(
        "--activation-percentile",
        type=float,
        default=100.0,
        help="Percentile of absolute activations used for int8 calibration.",
    )
    return parser.parse_args()


def load_calibration_windows(
    args: argparse.Namespace, checkpoint: dict
) -> torch.Tensor:
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
    x, y, metadata = build_windows(
        args.data_root, window_config, args.max_calibration_windows
    )
    split = subject_holdout_split(x, y, metadata)
    split = apply_task(split, config["task"])
    if config["normalization"] == "train":
        split = apply_train_normalization(split)
    return torch.from_numpy(split.x_train[: args.max_calibration_windows])


def calibrate_scales(
    model: torch.nn.Module,
    x: torch.Tensor,
    activation_percentile: float,
) -> dict[str, float]:
    with torch.no_grad():
        layers = model.net
        conv1 = layers[2](layers[1](layers[0](x)))
        pool1 = layers[3](conv1)
        conv2 = layers[6](layers[5](layers[4](pool1)))
        pool2 = layers[7](conv2)
        conv3 = layers[10](layers[9](layers[8](pool2)))
        gap = layers[12](layers[11](conv3))
        logits = layers[13](gap)
    return {
        "input": symmetric_scale(x, activation_percentile),
        "conv1": symmetric_scale(conv1, activation_percentile),
        "conv2": symmetric_scale(conv2, activation_percentile),
        "conv3": symmetric_scale(conv3, activation_percentile),
        "fc": symmetric_scale(logits, activation_percentile),
    }


def symmetric_scale(tensor: torch.Tensor, percentile: float) -> float:
    values = tensor.detach().abs().reshape(-1).cpu().numpy()
    max_abs = float(np.percentile(values, percentile))
    return max(max_abs / 127.0, 1.0 / 32768.0)


def extract_small_layers(
    model: torch.nn.Module, scales: dict[str, float]
) -> dict[str, dict]:
    layers = model.net
    conv1_w, conv1_b = fold_conv_bn(layers[0], layers[1])
    conv2_w, conv2_b = fold_conv_bn(layers[4], layers[5])
    conv3_w, conv3_b = fold_conv_bn(layers[8], layers[9])
    fc = layers[13]

    return {
        "conv1": quantize_conv(conv1_w, conv1_b, scales["input"], scales["conv1"]),
        "conv2": quantize_conv(conv2_w, conv2_b, scales["conv1"], scales["conv2"]),
        "conv3": quantize_conv(conv3_w, conv3_b, scales["conv2"], scales["conv3"]),
        "fc": quantize_linear(
            fc.weight.detach().cpu().numpy(),
            fc.bias.detach().cpu().numpy(),
            scales["conv3"],
            scales["fc"],
        ),
    }


def fold_conv_bn(
    conv: torch.nn.Conv1d, bn: torch.nn.BatchNorm1d
) -> tuple[np.ndarray, np.ndarray]:
    weight = conv.weight.detach().cpu()
    bias = (
        conv.bias.detach().cpu()
        if conv.bias is not None
        else torch.zeros(weight.shape[0])
    )
    scale = bn.weight.detach().cpu() / torch.sqrt(
        bn.running_var.detach().cpu() + bn.eps
    )
    folded_weight = weight * scale.reshape(-1, 1, 1)
    folded_bias = (
        bias - bn.running_mean.detach().cpu()
    ) * scale + bn.bias.detach().cpu()
    return folded_weight.numpy(), folded_bias.numpy()


def quantize_conv(
    weight_oik: np.ndarray,
    bias: np.ndarray,
    input_scale: float,
    output_scale: float,
) -> dict:
    out_ch, in_ch, kernel = weight_oik.shape
    weight_scales = np.maximum(
        np.max(np.abs(weight_oik), axis=(1, 2)) / 127.0, 1.0 / 32768.0
    )
    quant_weight = (
        np.round(weight_oik / weight_scales[:, None, None])
        .clip(-127, 127)
        .astype(np.int8)
    )
    # CMSIS-NN filter layout is [out_ch, kernel_h=1, kernel_w, in_ch].
    cmsis_weight = np.transpose(quant_weight, (0, 2, 1)).reshape(
        out_ch, 1, kernel, in_ch
    )
    quant_bias = np.round(bias / (input_scale * weight_scales)).astype(np.int32)
    multipliers, shifts = quantize_multiplier(
        input_scale * weight_scales / output_scale
    )
    return {
        "weight": cmsis_weight.reshape(-1),
        "bias": quant_bias,
        "mult": multipliers,
        "shift": shifts,
    }


def quantize_linear(
    weight_oi: np.ndarray,
    bias: np.ndarray,
    input_scale: float,
    output_scale: float,
) -> dict:
    max_abs = max(float(np.max(np.abs(weight_oi))), 1.0 / 32768.0)
    weight_scale = max_abs / 127.0
    quant_weight = np.round(weight_oi / weight_scale).clip(-127, 127).astype(np.int8)
    quant_bias = np.round(bias / (input_scale * weight_scale)).astype(np.int32)
    multiplier, shift = quantize_multiplier(
        np.asarray([input_scale * weight_scale / output_scale])
    )
    return {
        "weight": quant_weight.reshape(-1),
        "bias": quant_bias,
        "mult": int(multiplier[0]),
        "shift": int(shift[0]),
    }


def quantize_multiplier(real_multiplier: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    multipliers = []
    shifts = []
    for value in real_multiplier.astype(float):
        if value <= 0:
            multipliers.append(0)
            shifts.append(0)
            continue
        significand, exponent = math.frexp(value)
        q31 = int(round(significand * (1 << 31)))
        if q31 == (1 << 31):
            q31 //= 2
            exponent += 1
        multipliers.append(q31)
        shifts.append(exponent)
    return np.asarray(multipliers, dtype=np.int32), np.asarray(shifts, dtype=np.int32)


def extract_deployment_architecture(
    model: torch.nn.Module, checkpoint: dict
) -> dict[str, int]:
    layers = model.net
    return {
        "input_len": int(checkpoint["input_shape"][1]),
        "channels": int(checkpoint["input_shape"][0]),
        "classes": len(checkpoint["class_names"]),
        "conv1_out": int(layers[0].out_channels),
        "conv2_out": int(layers[4].out_channels),
        "conv3_out": int(layers[8].out_channels),
        "pool1_out_w": int(checkpoint["input_shape"][1]) // 2,
        "pool2_out_w": int(checkpoint["input_shape"][1]) // 4,
        "scratch_size": 32768,
    }


def render_header(
    checkpoint: dict,
    architecture: dict[str, int],
    scales: dict[str, float],
    layers: dict[str, dict],
) -> str:
    norm = checkpoint.get("normalization") or {
        "mean": [0.0] * checkpoint["input_shape"][0],
        "std": [1.0] * checkpoint["input_shape"][0],
    }
    class_names = checkpoint["class_names"]
    return "\n".join(
        [
            "#pragma once",
            "",
            "#include <stdint.h>",
            "",
            "// Generated by: uv run python -m model.export_cmsis",
            f'#define IMU_EXPORTED_MODEL_KIND "{checkpoint["model"]}"',
            f"#define IMU_CONV1_OUT_CH {architecture['conv1_out']}",
            f"#define IMU_CONV2_OUT_CH {architecture['conv2_out']}",
            f"#define IMU_CONV3_OUT_CH {architecture['conv3_out']}",
            f"#define IMU_POOL1_OUT_W {architecture['pool1_out_w']}",
            f"#define IMU_POOL2_OUT_W {architecture['pool2_out_w']}",
            f"#define IMU_MODEL_SCRATCH_SIZE {architecture['scratch_size']}",
            f"#define IMU_MODEL_INPUT_ZERO_POINT 0",
            f"#define IMU_MODEL_INPUT_SCALE {scales['input']:.10g}f",
            "",
            render_float_array("IMU_MODEL_NORM_MEAN", norm["mean"]),
            render_float_array("IMU_MODEL_NORM_STD", norm["std"]),
            render_class_names(class_names),
            render_layer("IMU_CONV1", layers["conv1"]),
            render_layer("IMU_CONV2", layers["conv2"]),
            render_layer("IMU_CONV3", layers["conv3"]),
            render_fc(layers["fc"]),
            "",
        ]
    )


def render_class_names(class_names: tuple[str, ...]) -> str:
    values = ", ".join(f'"{name}"' for name in class_names)
    return f"static const char *const IMU_MODEL_CLASS_NAMES[{len(class_names)}] = {{{values}}};\n"


def render_layer(prefix: str, layer: dict) -> str:
    return "\n".join(
        [
            render_int_array(f"{prefix}_W", layer["weight"], "int8_t"),
            render_int_array(f"{prefix}_B", layer["bias"], "int32_t"),
            render_int_array(f"{prefix}_MULT", layer["mult"], "int32_t"),
            render_int_array(f"{prefix}_SHIFT", layer["shift"], "int32_t"),
            "",
        ]
    )


def render_fc(layer: dict) -> str:
    return "\n".join(
        [
            render_int_array("IMU_FC_W", layer["weight"], "int8_t"),
            render_int_array("IMU_FC_B", layer["bias"], "int32_t"),
            f"#define IMU_FC_MULT {layer['mult']}",
            f"#define IMU_FC_SHIFT {layer['shift']}",
        ]
    )


def render_float_array(name: str, values: list[float]) -> str:
    body = ", ".join(f"{float(value):.10g}f" for value in values)
    return f"static const float {name}[{len(values)}] = {{{body}}};"


def render_int_array(name: str, values: np.ndarray, c_type: str) -> str:
    flat = [str(int(value)) for value in values.reshape(-1)]
    lines = []
    for idx in range(0, len(flat), 16):
        lines.append("    " + ", ".join(flat[idx : idx + 16]))
    body = ",\n".join(lines)
    return f"static const {c_type} {name}[{len(flat)}] = {{\n{body}\n}};"


if __name__ == "__main__":
    raise SystemExit(main())
