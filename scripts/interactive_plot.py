#!/usr/bin/env python3
"""Interactive IMU plotter for recorded CSV files.

Features:
- Trackpad/mouse scroll zoom (Shift = x-only, Ctrl = y-only).
- Right-click drag to pan.
- Toggle X/Y/Z axes.
- Highlight labeled regions (label == 1).
- Export current view to PNG.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib


@dataclass
class ImuData:
    timestamp: list[float]
    acc: list[tuple[float, float, float]]
    vel: list[tuple[float, float, float]]
    label: list[int]


@dataclass
class CsvMeta:
    test_type: str
    subject: str
    location: str


def parse_utc_timestamp(raw: str, row_number: int) -> datetime:
    raw_text = "" if raw is None else raw
    try:
        return datetime.fromisoformat(raw_text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"row {row_number}: invalid timestamp_utc {raw_text!r}"
        ) from exc


def parse_relative_timestamp_ms(raw: str, row_number: int) -> float:
    raw_text = "" if raw is None else raw
    try:
        return float(raw_text)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"row {row_number}: invalid timestamp value {raw_text!r}"
        ) from exc


def parse_csv_float(row: dict[str, str | None], field: str, row_number: int) -> float:
    raw = row.get(field, "")
    raw_text = "" if raw is None else raw
    try:
        return float(raw_text)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"row {row_number}: invalid {field} value {raw_text!r}"
        ) from exc


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


def load_csv(csv_path: Path) -> ImuData:
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("missing CSV header row")

        required_fields = {"acc_x", "acc_y", "acc_z", "vel_x", "vel_y", "vel_z"}
        missing_fields = [
            field for field in required_fields if field not in reader.fieldnames
        ]
        if missing_fields:
            raise ValueError(f"missing required columns: {', '.join(missing_fields)}")

        timestamp_field = None
        for field in ("timestamp", "timestamp_utc"):
            if field in reader.fieldnames:
                timestamp_field = field
                break
        if timestamp_field is None:
            raise ValueError(
                "missing required timestamp column: expected timestamp or timestamp_utc"
            )

        timestamps: list[float] = []
        acc: list[tuple[float, float, float]] = []
        vel: list[tuple[float, float, float]] = []
        labels: list[int] = []

        first_sample_time: datetime | None = None

        for row_number, row in enumerate(reader, start=2):
            if timestamp_field == "timestamp":
                sample_timestamp = (
                    parse_relative_timestamp_ms(row["timestamp"], row_number) / 1_000.0
                )
            else:
                sample_time = parse_utc_timestamp(row["timestamp_utc"], row_number)
                if first_sample_time is None:
                    first_sample_time = sample_time
                sample_timestamp = (sample_time - first_sample_time).total_seconds()

            timestamps.append(sample_timestamp)
            acc.append(
                (
                    parse_csv_float(row, "acc_x", row_number),
                    parse_csv_float(row, "acc_y", row_number),
                    parse_csv_float(row, "acc_z", row_number),
                )
            )
            vel.append(
                (
                    parse_csv_float(row, "vel_x", row_number),
                    parse_csv_float(row, "vel_y", row_number),
                    parse_csv_float(row, "vel_z", row_number),
                )
            )
            labels.append(parse_csv_label(row, row_number))

    return ImuData(timestamp=timestamps, acc=acc, vel=vel, label=labels)


def parse_csv_metadata(csv_path: Path) -> CsvMeta:
    test_type = csv_path.parent.name
    stem = csv_path.stem
    subject = ""
    location = ""
    if "_" in stem:
        last = stem.split("_")[-1]
    else:
        last = stem
    if "-" in last:
        subject, location = last.split("-", 1)
    else:
        subject = last
    return CsvMeta(test_type=test_type, subject=subject, location=location)


def build_title(meta: CsvMeta) -> str:
    parts = [meta.test_type, meta.subject, meta.location]
    safe = [part for part in parts if part]
    return " | ".join(safe) if safe else "IMU Plot"


def export_png(csv_path: Path, output_dir: Path | None = None) -> Path:
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    data = load_csv(csv_path)
    meta = parse_csv_metadata(csv_path)
    title = build_title(meta)

    times = data.timestamp
    xs = [row[0] for row in data.acc]
    ys = [row[1] for row in data.acc]
    zs = [row[2] for row in data.acc]

    fig = Figure(figsize=(8, 5), dpi=150)
    ax = fig.add_subplot(111)

    ax.plot(times, xs, label="x", color="#ff6b6b", linewidth=1.5)
    ax.plot(times, ys, label="y", color="#ffd43b", linewidth=1.5)
    ax.plot(times, zs, label="z", color="#69db7c", linewidth=1.5)

    for left, right in compute_label_regions(times, data.label):
        ax.axvspan(left, right, color="#7cfc9a", alpha=0.25)

    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("acc")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "screenshots"

    filename = "_".join([meta.test_type, meta.subject, meta.location, "acc"]) + ".png"
    output_path = output_dir / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return output_path


def compute_label_regions(
    timestamps: list[float], labels: list[int]
) -> list[tuple[float, float]]:
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
        prev_left, prev_right = merged[-1]
        if left <= prev_right:
            merged[-1] = (prev_left, max(prev_right, right))
        else:
            merged.append((left, right))
    return merged


class InteractivePlot:
    COLORS = {
        "x": "#ff6b6b",
        "y": "#ffd43b",
        "z": "#69db7c",
    }

    def __init__(
        self, csv_files: list[Path], title: str, output_dir: Path | None
    ) -> None:
        if not csv_files:
            raise ValueError("no CSV files provided")

        self.csv_files = csv_files
        self.output_dir = output_dir
        self.current_index = 0
        self.data = load_csv(self.csv_files[self.current_index])
        meta = parse_csv_metadata(self.csv_files[self.current_index])
        self.title = build_title(meta) if title == "" else title
        self.root = tk.Tk()
        self.root.title("IMU Interactive Plot")
        self.root.geometry("1280x820")

        self.series_var = tk.StringVar(value="acc")
        self.show_x = tk.BooleanVar(value=True)
        self.show_y = tk.BooleanVar(value=True)
        self.show_z = tk.BooleanVar(value=True)
        self._drag_start: tuple[float, float] | None = None

        self._build_ui()
        self._draw_plot(reset_view=True)

    def _build_ui(self) -> None:
        matplotlib.use("TkAgg")
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        toolbar = ttk.Frame(self.root)
        toolbar.pack(fill="x", padx=12, pady=8)

        ttk.Checkbutton(
            toolbar, text="X", variable=self.show_x, command=self._draw_plot
        ).pack(side="left")
        ttk.Checkbutton(
            toolbar, text="Y", variable=self.show_y, command=self._draw_plot
        ).pack(side="left")
        ttk.Checkbutton(
            toolbar, text="Z", variable=self.show_z, command=self._draw_plot
        ).pack(side="left")

        ttk.Button(
            toolbar, text="Reset View", command=lambda: self._draw_plot(reset_view=True)
        ).pack(side="right", padx=6)
        ttk.Button(toolbar, text="Prev", command=self._prev_file).pack(
            side="right", padx=6
        )
        ttk.Button(toolbar, text="Next", command=self._next_file).pack(
            side="right", padx=6
        )
        ttk.Button(toolbar, text="Export PNG", command=self._export_png).pack(
            side="right"
        )

        self.file_var = tk.StringVar(value=self._file_status())
        ttk.Label(self.root, textvariable=self.file_var).pack(
            anchor="w", padx=12, pady=(0, 6)
        )

        self.figure = Figure(figsize=(8, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("key_press_event", self._on_key)

    def _current_series(self) -> tuple[list[float], list[float], list[float]]:
        series = self.data.acc
        xs = [row[0] for row in series]
        ys = [row[1] for row in series]
        zs = [row[2] for row in series]
        return xs, ys, zs

    def _draw_plot(self, reset_view: bool = False) -> None:
        self.ax.clear()
        times = self.data.timestamp
        xs, ys, zs = self._current_series()

        if self.show_x.get():
            self.ax.plot(times, xs, label="x", color=self.COLORS["x"], linewidth=1.5)
        if self.show_y.get():
            self.ax.plot(times, ys, label="y", color=self.COLORS["y"], linewidth=1.5)
        if self.show_z.get():
            self.ax.plot(times, zs, label="z", color=self.COLORS["z"], linewidth=1.5)

        for left, right in compute_label_regions(times, self.data.label):
            self.ax.axvspan(left, right, color="#7cfc9a", alpha=0.25)

        self.ax.set_title(self.title)
        self.ax.set_xlabel("time (s)")
        self.ax.set_ylabel("acc")
        self.ax.grid(True, alpha=0.25)

        if any([self.show_x.get(), self.show_y.get(), self.show_z.get()]):
            self.ax.legend(loc="upper right")

        if reset_view:
            self.ax.relim()
            self.ax.autoscale()

        self.canvas.draw_idle()

    def _export_png(self) -> None:
        default_name = self._default_png_name()
        initial_dir = str(self._default_output_dir())
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
            initialfile=default_name,
            initialdir=initial_dir,
            title="Export plot as PNG",
        )
        if not path:
            return
        self._save_png(Path(path))

    def _on_scroll(self, event) -> None:
        if event.xdata is None or event.ydata is None:
            return

        dx, dy = self._scroll_deltas(event)
        if dx == 0 and dy == 0:
            return

        self._pan(dx, dy)
        self.canvas.draw_idle()

    def _zoom_x(self, center: float, scale: float) -> None:
        left, right = self.ax.get_xlim()
        span = (right - left) * scale
        self.ax.set_xlim(center - span / 2.0, center + span / 2.0)

    def _zoom_y(self, center: float, scale: float) -> None:
        bottom, top = self.ax.get_ylim()
        span = (top - bottom) * scale
        self.ax.set_ylim(center - span / 2.0, center + span / 2.0)

    def _on_press(self, event) -> None:
        if event.button != 3 or event.xdata is None or event.ydata is None:
            return
        self._drag_start = (event.xdata, event.ydata)

    def _on_release(self, event) -> None:
        self._drag_start = None

    def _on_motion(self, event) -> None:
        if self._drag_start is None or event.xdata is None or event.ydata is None:
            return
        start_x, start_y = self._drag_start
        dx = start_x - event.xdata
        dy = start_y - event.ydata

        left, right = self.ax.get_xlim()
        bottom, top = self.ax.get_ylim()
        self.ax.set_xlim(left + dx, right + dx)
        self.ax.set_ylim(bottom + dy, top + dy)
        self.canvas.draw_idle()

    def run(self) -> None:
        self.root.mainloop()

    def _default_output_dir(self) -> Path:
        if self.output_dir is None:
            return self.csv_files[self.current_index].parent / "screenshots"
        return self.output_dir

    def _default_png_name(self) -> str:
        meta = parse_csv_metadata(self.csv_files[self.current_index])
        parts = [meta.test_type, meta.subject, meta.location, "acc"]
        safe = [part for part in parts if part]
        return "_".join(safe) + ".png"

    def _save_png(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(path, dpi=200, bbox_inches="tight")

    def _file_status(self) -> str:
        return f"file {self.current_index + 1} of {len(self.csv_files)}: {self.csv_files[self.current_index].name}"

    def _load_index(self, index: int) -> None:
        self.current_index = max(0, min(index, len(self.csv_files) - 1))
        self.data = load_csv(self.csv_files[self.current_index])
        meta = parse_csv_metadata(self.csv_files[self.current_index])
        self.title = build_title(meta)
        self.file_var.set(self._file_status())
        self._draw_plot(reset_view=True)

    def _prompt_export_current(self) -> None:
        name = self._default_png_name()
        answer = messagebox.askyesno(
            title="Export PNG",
            message=f"Export PNG for {self.csv_files[self.current_index].name} as {name}?",
        )
        if answer:
            path = self._default_output_dir() / name
            self._save_png(path)

    def _next_file(self) -> None:
        self._prompt_export_current()
        if self.current_index < len(self.csv_files) - 1:
            self._load_index(self.current_index + 1)

    def _prev_file(self) -> None:
        if self.current_index > 0:
            self._load_index(self.current_index - 1)

    def _scroll_deltas(self, event) -> tuple[float, float]:
        dx = 0.0
        dy = 0.0

        gui_event = getattr(event, "guiEvent", None)
        if gui_event is not None:
            if hasattr(gui_event, "delta_x") and hasattr(gui_event, "delta_y"):
                dx = float(gui_event.delta_x)
                dy = float(gui_event.delta_y)
            elif hasattr(gui_event, "delta"):
                delta = gui_event.delta
                if isinstance(delta, tuple) and len(delta) == 2:
                    dx = float(delta[0])
                    dy = float(delta[1])
                else:
                    dy = float(delta)

        if dx == 0.0 and dy == 0.0:
            dy = float(getattr(event, "step", 0.0))

        if event.key == "shift" and dx == 0.0:
            dx, dy = dy, 0.0

        return dx, dy

    def _pan(self, dx: float, dy: float) -> None:
        left, right = self.ax.get_xlim()
        bottom, top = self.ax.get_ylim()
        xspan = right - left
        yspan = top - bottom

        # Normalize scroll deltas to a reasonable pan amount.
        scale = 0.0025
        self.ax.set_xlim(left - dx * scale * xspan, right - dx * scale * xspan)
        self.ax.set_ylim(bottom + dy * scale * yspan, top + dy * scale * yspan)

    def _on_key(self, event) -> None:
        if event.key in ("+", "="):
            self._zoom_x(0.5 * sum(self.ax.get_xlim()), 0.85)
            self._zoom_y(0.5 * sum(self.ax.get_ylim()), 0.85)
            self.canvas.draw_idle()
        elif event.key in ("-", "_"):
            self._zoom_x(0.5 * sum(self.ax.get_xlim()), 1.15)
            self._zoom_y(0.5 * sum(self.ax.get_ylim()), 1.15)
            self.canvas.draw_idle()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive plotter for IMU CSV recordings."
    )
    parser.add_argument("--csv", help="path to CSV file")
    parser.add_argument("--csv-dir", help="directory of CSV files to step through")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="scan for CSVs recursively under --csv-dir",
    )
    parser.add_argument(
        "--out-dir", help="directory to save exported PNGs (default: CSV folder)"
    )
    parser.add_argument(
        "--batch", action="store_true", help="export PNGs without opening a window"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    csv_files: list[Path] = []
    if args.csv_dir:
        csv_dir = Path(args.csv_dir).expanduser().resolve()
        if args.recursive:
            csv_files = sorted(csv_dir.rglob("*.csv"))
        else:
            csv_files = sorted(csv_dir.glob("*.csv"))
    elif args.csv:
        csv_files = [Path(args.csv).expanduser().resolve()]

    if not csv_files:
        raise ValueError("no CSV files provided; use --csv or --csv-dir")

    output_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else repo_root / "screenshots"
    )
    if args.batch:
        for csv_path in csv_files:
            export_png(csv_path, output_dir=output_dir)
    else:
        app = InteractivePlot(csv_files=csv_files, title="", output_dir=output_dir)
        app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
