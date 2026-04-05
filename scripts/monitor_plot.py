#!/usr/bin/env python3
"""Live IMU visualizer fed by `cargo monitor`.

This uses only Python's standard library. The graphing is drawn directly on a
tkinter Canvas rather than using a third-party plotting package.
"""

from __future__ import annotations

import argparse
import csv
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import queue
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import tkinter as tk

DATA_LINE_RE = re.compile(r"acc=\[(?P<acc>[^\]]+)\]\s+velocity=\[(?P<vel>[^\]]+)\]")
BACKGROUND = "#0b0f14"
PANEL_BG = "#111924"
GRID = "#223043"
TEXT = "#d9e2f2"
MUTED = "#7f8da3"
ACC_COLORS = ("#ff6b6b", "#ff922b", "#ffd43b")
VEL_COLORS = ("#4dabf7", "#22b8cf", "#69db7c")
LABEL_SHADE = "#7cfc9a"
LABEL_SHADE_STIPPLE = "gray25"


@dataclass
class Sample:
    timestamp: float
    acceleration: tuple[float, float, float]
    velocity: tuple[float, float, float]
    label: int = 0


class MonitorReader(threading.Thread):
    def __init__(
        self,
        command: list[str],
        workdir: Path,
        output_queue: queue.Queue[tuple[str, object]],
    ) -> None:
        super().__init__(daemon=True)
        self.command = command
        self.source_name = "cargo monitor"
        self.replace_status_on_success_exit = True
        self.workdir = workdir
        self.output_queue = output_queue
        self.stop_event = threading.Event()
        self.process: subprocess.Popen[str] | None = None
        self.first_sample_time: float | None = None

    def run(self) -> None:
        try:
            self.process = subprocess.Popen(
                self.command,
                cwd=self.workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            self.output_queue.put(("error", f"failed to launch {' '.join(self.command)}: {exc}"))
            return

        assert self.process.stdout is not None

        for raw_line in self.process.stdout:
            if self.stop_event.is_set():
                break

            line = raw_line.strip()
            if not line:
                continue

            sample = self._parse_sample(line)
            if sample is not None:
                self.output_queue.put(("sample", sample))
            else:
                self.output_queue.put(("status", line))

        return_code = self.process.wait()
        if not self.stop_event.is_set():
            self.output_queue.put(("exit", return_code))

    def stop(self) -> None:
        self.stop_event.set()
        if self.process is None or self.process.poll() is not None:
            return

        try:
            self.process.send_signal(signal.SIGINT)
            self.process.wait(timeout=1.5)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            self.process.terminate()
            try:
                self.process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def _parse_sample(self, line: str) -> Sample | None:
        match = DATA_LINE_RE.search(line)
        if match is None:
            return None

        acceleration = self._parse_vector(match.group("acc"))
        velocity = self._parse_vector(match.group("vel"))
        if acceleration is None or velocity is None:
            self.output_queue.put(("status", f"ignored malformed sample: {line}"))
            return None

        now = time.monotonic()
        if self.first_sample_time is None:
            self.first_sample_time = now

        return Sample(
            timestamp=now - self.first_sample_time,
            acceleration=acceleration,
            velocity=velocity,
            label=0,
        )

    @staticmethod
    def _parse_vector(raw: str) -> tuple[float, float, float] | None:
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 3:
            return None

        try:
            return float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            return None


class CsvReader(threading.Thread):
    VALUE_FIELDS = ("acc_x", "acc_y", "acc_z", "vel_x", "vel_y", "vel_z")
    TIMESTAMP_FIELDS = ("timestamp", "timestamp_utc")

    def __init__(self, csv_path: Path, output_queue: queue.Queue[tuple[str, object]]) -> None:
        super().__init__(daemon=True)
        self.csv_path = csv_path
        self.output_queue = output_queue
        self.stop_event = threading.Event()
        self.command = ["csv", str(csv_path)]
        self.source_name = "CSV loader"
        self.replace_status_on_success_exit = False

    def run(self) -> None:
        try:
            samples_loaded = self._load_csv()
        except (OSError, ValueError, csv.Error) as exc:
            self.output_queue.put(("error", f"failed to load CSV {self.csv_path}: {exc}"))
            self.output_queue.put(("exit", 1))
            return

        if not self.stop_event.is_set():
            self.output_queue.put(("status", f"loaded {samples_loaded} samples from {self.csv_path}"))
            self.output_queue.put(("exit", 0))

    def stop(self) -> None:
        self.stop_event.set()

    def _load_csv(self) -> int:
        with self.csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames is None:
                raise ValueError("missing CSV header row")

            missing_fields = [field for field in self.VALUE_FIELDS if field not in reader.fieldnames]
            if missing_fields:
                raise ValueError(f"missing required columns: {', '.join(missing_fields)}")

            timestamp_field = next((field for field in self.TIMESTAMP_FIELDS if field in reader.fieldnames), None)
            if timestamp_field is None:
                raise ValueError(
                    "missing required timestamp column: expected one of "
                    f"{', '.join(self.TIMESTAMP_FIELDS)}"
                )

            samples_loaded = 0
            first_sample_time: datetime | None = None

            for row_number, row in enumerate(reader, start=2):
                if self.stop_event.is_set():
                    break

                if timestamp_field == "timestamp":
                    sample_timestamp = parse_relative_timestamp_ms(row["timestamp"], row_number) / 1_000.0
                else:
                    sample_time = parse_utc_timestamp(row["timestamp_utc"], row_number)
                    if first_sample_time is None:
                        first_sample_time = sample_time
                    sample_timestamp = (sample_time - first_sample_time).total_seconds()

                sample = Sample(
                    timestamp=sample_timestamp,
                    acceleration=(
                        parse_csv_float(row, "acc_x", row_number),
                        parse_csv_float(row, "acc_y", row_number),
                        parse_csv_float(row, "acc_z", row_number),
                    ),
                    velocity=(
                        parse_csv_float(row, "vel_x", row_number),
                        parse_csv_float(row, "vel_y", row_number),
                        parse_csv_float(row, "vel_z", row_number),
                    ),
                    label=parse_csv_label(row, row_number),
                )
                self.output_queue.put(("sample", sample))
                samples_loaded += 1

        return samples_loaded


class LivePlotApp:
    def __init__(self, reader: MonitorReader | CsvReader, history_size: int, refresh_ms: int) -> None:
        self.reader = reader
        self.history_size = history_size
        self.refresh_ms = refresh_ms
        self.events: queue.Queue[tuple[str, object]] = reader.output_queue

        self.timestamps: deque[float] = deque(maxlen=history_size)
        self.acceleration = [deque(maxlen=history_size) for _ in range(3)]
        self.velocity = [deque(maxlen=history_size) for _ in range(3)]
        self.labels: deque[int] = deque(maxlen=history_size)
        self.samples_seen = 0
        self.last_status = f"waiting for samples from {self.reader.source_name}"
        self.last_exit_code: int | None = None

        self.root = tk.Tk()
        self.root.title("MKBOXPRO IMU Live Plot")
        self.root.geometry("1280x820")
        self.root.configure(bg=BACKGROUND)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.status_var = tk.StringVar(value=self._build_status_text())
        self.status_label = tk.Label(
            self.root,
            textvariable=self.status_var,
            bg=BACKGROUND,
            fg=TEXT,
            anchor="w",
            justify="left",
            padx=16,
            pady=12,
            font=("Menlo", 12),
        )
        self.status_label.pack(fill="x")

        self.canvas = tk.Canvas(self.root, bg=BACKGROUND, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

    def run(self) -> None:
        self.reader.start()
        self.root.after(self.refresh_ms, self.refresh)
        self.root.mainloop()

    def refresh(self) -> None:
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break

            if kind == "sample":
                sample = payload
                assert isinstance(sample, Sample)
                self._append_sample(sample)
            elif kind == "status":
                self.last_status = str(payload)
            elif kind == "error":
                self.last_status = str(payload)
            elif kind == "exit":
                self.last_exit_code = int(payload)
                if self.last_exit_code != 0 or self.reader.replace_status_on_success_exit:
                    self.last_status = f"{self.reader.source_name} exited with code {self.last_exit_code}"

        self.status_var.set(self._build_status_text())
        self.redraw()
        self.root.after(self.refresh_ms, self.refresh)

    def close(self) -> None:
        self.reader.stop()
        self.root.destroy()

    def _append_sample(self, sample: Sample) -> None:
        self.timestamps.append(sample.timestamp)
        for axis, value in enumerate(sample.acceleration):
            self.acceleration[axis].append(value)
        for axis, value in enumerate(sample.velocity):
            self.velocity[axis].append(value)
        self.labels.append(sample.label)
        self.samples_seen += 1

    def _build_status_text(self) -> str:
        command_text = " ".join(self.reader.command)
        sample_text = f"samples: {self.samples_seen}"
        if self.last_exit_code is None:
            state_text = "state: running"
        else:
            state_text = f"state: exited ({self.last_exit_code})"
        return (
            f"command: {command_text}\n"
            f"{sample_text}    {state_text}\n"
            f"last message: {self.last_status}"
        )

    def redraw(self) -> None:
        self.canvas.delete("all")
        width = max(self.canvas.winfo_width(), 800)
        height = max(self.canvas.winfo_height(), 500)

        outer_pad = 18
        gap = 18
        panel_height = (height - outer_pad * 2 - gap) / 2

        self._draw_panel(
            title="Acceleration (m/s^2)",
            x0=outer_pad,
            y0=outer_pad,
            x1=width - outer_pad,
            y1=outer_pad + panel_height,
            series=self.acceleration,
            colors=ACC_COLORS,
            axis_labels=("ax", "ay", "az"),
        )
        self._draw_panel(
            title="Velocity (m/s)",
            x0=outer_pad,
            y0=outer_pad + panel_height + gap,
            x1=width - outer_pad,
            y1=height - outer_pad,
            series=self.velocity,
            colors=VEL_COLORS,
            axis_labels=("vx", "vy", "vz"),
        )

    def _draw_panel(
        self,
        title: str,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        series: list[deque[float]],
        colors: tuple[str, str, str],
        axis_labels: tuple[str, str, str],
    ) -> None:
        self.canvas.create_rectangle(x0, y0, x1, y1, fill=PANEL_BG, outline=GRID, width=1)
        self.canvas.create_text(
            x0 + 18,
            y0 + 18,
            anchor="w",
            text=title,
            fill=TEXT,
            font=("Menlo", 15, "bold"),
        )

        if len(self.timestamps) < 2:
            self.canvas.create_text(
                (x0 + x1) / 2,
                (y0 + y1) / 2,
                text="Waiting for parsed IMU samples...",
                fill=MUTED,
                font=("Menlo", 14),
            )
            return

        plot_left = x0 + 70
        plot_right = x1 - 18
        plot_top = y0 + 42
        plot_bottom = y1 - 34
        plot_width = plot_right - plot_left
        plot_height = plot_bottom - plot_top

        scale = self._compute_scale(series)
        start_time = self.timestamps[0]
        end_time = self.timestamps[-1]
        span = max(end_time - start_time, 1e-6)
        self._draw_label_regions(plot_left, plot_top, plot_right, plot_bottom, start_time, span)
        self._draw_grid(plot_left, plot_top, plot_right, plot_bottom, scale)

        for label, values, color in zip(axis_labels, series, colors):
            points: list[float] = []
            for sample_time, value in zip(self.timestamps, values):
                x = plot_left + ((sample_time - start_time) / span) * plot_width
                y = plot_top + ((scale - value) / (2 * scale)) * plot_height
                points.extend((x, y))

            if len(points) >= 4:
                self.canvas.create_line(*points, fill=color, width=2)
                last_x = points[-2]
                last_y = points[-1]
                self.canvas.create_oval(
                    last_x - 3,
                    last_y - 3,
                    last_x + 3,
                    last_y + 3,
                    fill=color,
                    outline="",
                )

            latest_value = values[-1]
            legend_index = axis_labels.index(label)
            self.canvas.create_text(
                plot_right - 8,
                y0 + 18 + legend_index * 18,
                anchor="e",
                text=f"{label} {latest_value:+.3f}",
                fill=color,
                font=("Menlo", 12),
            )

        if any(self.labels):
            legend_y = y0 + 18 + len(axis_labels) * 18
            self.canvas.create_rectangle(
                plot_right - 84,
                legend_y - 8,
                plot_right - 70,
                legend_y + 6,
                fill=LABEL_SHADE,
                outline="",
                stipple=LABEL_SHADE_STIPPLE,
            )
            self.canvas.create_text(
                plot_right - 8,
                legend_y,
                anchor="e",
                text="flag",
                fill=MUTED,
                font=("Menlo", 12),
            )

        self.canvas.create_text(
            plot_left,
            y1 - 12,
            anchor="w",
            text=f"{start_time:.1f}s",
            fill=MUTED,
            font=("Menlo", 11),
        )
        self.canvas.create_text(
            plot_right,
            y1 - 12,
            anchor="e",
            text=f"{end_time:.1f}s",
            fill=MUTED,
            font=("Menlo", 11),
        )

    def _draw_grid(self, left: float, top: float, right: float, bottom: float, scale: float) -> None:
        height = bottom - top
        width = right - left

        for idx in range(5):
            fraction = idx / 4
            x = left + fraction * width
            self.canvas.create_line(x, top, x, bottom, fill=GRID)

        for ratio in (-1.0, -0.5, 0.0, 0.5, 1.0):
            y = top + ((1.0 - (ratio + 1.0) / 2.0) * height)
            color = TEXT if ratio == 0.0 else GRID
            self.canvas.create_line(left, y, right, y, fill=color)
            self.canvas.create_text(
                left - 8,
                y,
                anchor="e",
                text=f"{ratio * scale:+.2f}",
                fill=MUTED,
                font=("Menlo", 11),
            )

    def _draw_label_regions(
        self,
        plot_left: float,
        plot_top: float,
        plot_right: float,
        plot_bottom: float,
        start_time: float,
        span: float,
    ) -> None:
        timestamps = list(self.timestamps)
        labels = list(self.labels)
        if len(timestamps) < 2 or not any(labels):
            return

        for region_start, region_end in self._compute_labeled_regions(timestamps, labels):
            x0 = plot_left + ((region_start - start_time) / span) * (plot_right - plot_left)
            x1 = plot_left + ((region_end - start_time) / span) * (plot_right - plot_left)
            self.canvas.create_rectangle(
                x0,
                plot_top,
                x1,
                plot_bottom,
                fill=LABEL_SHADE,
                outline="",
                stipple=LABEL_SHADE_STIPPLE,
            )

    @staticmethod
    def _compute_labeled_regions(timestamps: list[float], labels: list[int]) -> list[tuple[float, float]]:
        if len(timestamps) != len(labels) or len(timestamps) < 2:
            return []

        boundaries: list[tuple[float, float]] = []
        end_time = timestamps[-1]

        for index, (timestamp, label) in enumerate(zip(timestamps, labels)):
            if label != 1:
                continue

            if index == 0:
                left = timestamps[0]
            else:
                left = (timestamps[index - 1] + timestamp) / 2.0

            if index == len(timestamps) - 1:
                right = end_time
            else:
                right = (timestamp + timestamps[index + 1]) / 2.0

            boundaries.append((left, right))

        if not boundaries:
            return []

        merged = [boundaries[0]]
        for left, right in boundaries[1:]:
            previous_left, previous_right = merged[-1]
            if left <= previous_right:
                merged[-1] = (previous_left, max(previous_right, right))
            else:
                merged.append((left, right))
        return merged

    @staticmethod
    def _compute_scale(series: list[deque[float]]) -> float:
        max_value = 0.1
        for values in series:
            for value in values:
                max_value = max(max_value, abs(value))
        return max_value * 1.1


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run `cargo monitor` and draw custom live plots for acceleration and velocity.",
    )
    parser.add_argument(
        "--command",
        default="cargo monitor",
        help="command to launch the serial monitor (default: %(default)s)",
    )
    parser.add_argument(
        "--csv",
        help="load samples from a CSV file instead of launching the monitor command",
    )
    parser.add_argument(
        "--cwd",
        default=str(repo_root),
        help="working directory for the monitor command (default: repository root)",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=500,
        help="number of recent samples to keep on screen",
    )
    parser.add_argument(
        "--refresh-ms",
        type=int,
        default=33,
        help="UI refresh period in milliseconds",
    )
    return parser.parse_args()


def parse_utc_timestamp(raw: str, row_number: int) -> datetime:
    raw_text = "" if raw is None else raw
    try:
        return datetime.fromisoformat(raw_text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"row {row_number}: invalid timestamp_utc {raw_text!r}") from exc


def parse_relative_timestamp_ms(raw: str, row_number: int) -> float:
    raw_text = "" if raw is None else raw
    try:
        return float(raw_text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {row_number}: invalid timestamp value {raw_text!r}") from exc


def parse_csv_float(row: dict[str, str | None], field: str, row_number: int) -> float:
    raw = row.get(field, "")
    raw_text = "" if raw is None else raw
    try:
        return float(raw_text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {row_number}: invalid {field} value {raw_text!r}") from exc


def parse_csv_label(row: dict[str, str | None], row_number: int) -> int:
    raw = row.get("label", "")
    raw_text = "" if raw is None else raw.strip()
    if raw_text == "":
        return 0

    try:
        label = int(raw_text)
    except ValueError as exc:
        raise ValueError(f"row {row_number}: invalid label value {raw_text!r}") from exc

    if label not in (0, 1):
        raise ValueError(f"row {row_number}: label must be 0 or 1, got {raw_text!r}")

    return label


def main() -> int:
    args = parse_args()

    reader_queue: queue.Queue[tuple[str, object]] = queue.Queue()
    workdir = Path(args.cwd)

    if args.csv:
        csv_path = Path(args.csv).expanduser()
        if not csv_path.is_absolute():
            csv_path = workdir / csv_path
        reader = CsvReader(csv_path=csv_path.resolve(), output_queue=reader_queue)
    else:
        command = shlex.split(args.command)
        if not command:
            print("error: --command must not be empty", file=sys.stderr)
            return 2
        reader = MonitorReader(command=command, workdir=workdir, output_queue=reader_queue)

    app = LivePlotApp(reader=reader, history_size=max(10, args.history), refresh_ms=max(10, args.refresh_ms))

    try:
        app.run()
    except KeyboardInterrupt:
        reader.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
