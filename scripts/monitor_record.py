#!/usr/bin/env python3
"""Record labeled IMU samples from `cargo monitor` into a CSV file.

The firmware currently streams samples as lines shaped like:

    acc=[ax,ay,az] velocity=[vx,vy,vz]

This recorder launches the monitor command, parses those lines, and writes CSV
rows with a host-side relative timestamp in milliseconds plus a live binary label.

Interactive controls:
    1 + Enter    start a labeled region
    0 + Enter    end a labeled region
    q + Enter    stop recording and close the CSV
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from typing import Callable, TextIO

DATA_LINE_RE = re.compile(r"acc=\[(?P<acc>[^\]]+)\]\s+velocity=\[(?P<vel>[^\]]+)\]")


@dataclass(frozen=True)
class Sample:
    timestamp_ms: int
    acceleration: tuple[float, float, float]
    velocity: tuple[float, float, float]


class LabelTimeline:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._change_times_ms = [0]
        self._change_values = [0]
        self._stop_event = threading.Event()

    def set_label(self, value: int) -> tuple[bool, int]:
        if value not in (0, 1):
            raise ValueError(f"label must be 0 or 1, got {value}")

        now_ms = monotonic_ms()
        with self._lock:
            if self._change_values[-1] == value:
                return False, now_ms
            self._change_times_ms.append(now_ms)
            self._change_values.append(value)
            return True, now_ms

    def label_for(self, timestamp_ms: int) -> int:
        with self._lock:
            index = bisect_right(self._change_times_ms, timestamp_ms) - 1
            return self._change_values[index]

    def current_label(self) -> int:
        with self._lock:
            return self._change_values[-1]

    def request_stop(self) -> None:
        self._stop_event.set()

    def stop_requested(self) -> bool:
        return self._stop_event.is_set()


class ControlReader(threading.Thread):
    def __init__(
        self,
        timeline: LabelTimeline,
        stop_monitor: Callable[[], None],
        input_stream: TextIO,
        output_stream: TextIO,
    ) -> None:
        super().__init__(daemon=True)
        self.timeline = timeline
        self.stop_monitor = stop_monitor
        self.input_stream = input_stream
        self.output_stream = output_stream

    def run(self) -> None:
        self._write_line("controls: 1=start label, 0=stop label, q=quit")
        self._write_line(f"label state: {self.timeline.current_label()}")

        while not self.timeline.stop_requested():
            line = self.input_stream.readline()
            if line == "":
                return

            command = line.strip().lower()
            if not command:
                continue

            if command in {"1", "start", "on"}:
                changed, at_ms = self.timeline.set_label(1)
                state = "started" if changed else "already active"
                self._write_line(f"label=1 {state} at {format_monotonic_ms(at_ms)}")
                continue

            if command in {"0", "stop", "off"}:
                changed, at_ms = self.timeline.set_label(0)
                state = "stopped" if changed else "already inactive"
                self._write_line(f"label=0 {state} at {format_monotonic_ms(at_ms)}")
                continue

            if command in {"q", "quit", "exit"}:
                self.timeline.request_stop()
                self._write_line("stop requested, closing recorder")
                self.stop_monitor()
                return

            self._write_line("unknown command: use 1, 0, or q")

    def _write_line(self, message: str) -> None:
        print(message, file=self.output_stream, flush=True)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run `cargo monitor` and record labeled IMU samples into a CSV file.",
    )
    parser.add_argument(
        "name",
        help="custom name included in the output filename",
    )
    parser.add_argument(
        "--command",
        default="cargo monitor",
        help="command used to launch the monitor stream (default: %(default)s)",
    )
    parser.add_argument(
        "--cwd",
        default=str(repo_root),
        help="working directory for the monitor command (default: repository root)",
    )
    parser.add_argument(
        "--output-dir",
        default="recordings",
        help="directory where the CSV file will be written (default: %(default)s)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="stop automatically after this many parsed samples (default: run until stopped)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    command = shlex.split(args.command)
    if not command:
        print("error: --command must not be empty", file=sys.stderr)
        return 2

    repo_cwd = Path(args.cwd).expanduser().resolve()
    output_dir = resolve_output_dir(repo_cwd, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        filename = f"{utc_filename_stamp()}_{slugify_filename(args.name)}.csv"
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    output_path = output_dir / filename
    if output_path.exists():
        print(f"error: output file already exists: {output_path}", file=sys.stderr)
        return 2

    try:
        process = subprocess.Popen(
            command,
            cwd=repo_cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        print(f"error: failed to launch {' '.join(command)}: {exc}", file=sys.stderr)
        return 1

    timeline = LabelTimeline()

    def stop_monitor() -> None:
        stop_process(process)

    controls = ControlReader(
        timeline=timeline,
        stop_monitor=stop_monitor,
        input_stream=sys.stdin,
        output_stream=sys.stderr,
    )
    controls.start()

    print(f"writing CSV to {output_path}", file=sys.stderr, flush=True)
    print(f"monitor command: {' '.join(command)}", file=sys.stderr, flush=True)

    samples_written = 0
    first_sample_ms: int | None = None

    try:
        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "timestamp",
                    "acc_x",
                    "acc_y",
                    "acc_z",
                    "vel_x",
                    "vel_y",
                    "vel_z",
                    "label",
                ]
            )

            assert process.stdout is not None
            for raw_line in process.stdout:
                line = raw_line.strip()
                if not line:
                    continue

                sample = parse_sample(line)
                if sample is None:
                    print(line, file=sys.stderr, flush=True)
                    if timeline.stop_requested():
                        break
                    continue

                label = timeline.label_for(sample.timestamp_ms)
                if first_sample_ms is None:
                    first_sample_ms = sample.timestamp_ms

                writer.writerow(
                    [
                        relative_timestamp_ms(sample.timestamp_ms, first_sample_ms),
                        sample.acceleration[0],
                        sample.acceleration[1],
                        sample.acceleration[2],
                        sample.velocity[0],
                        sample.velocity[1],
                        sample.velocity[2],
                        label,
                    ]
                )
                csv_file.flush()
                samples_written += 1

                if args.max_samples > 0 and samples_written >= args.max_samples:
                    timeline.request_stop()
                    stop_monitor()
                    break

                if timeline.stop_requested():
                    break
    except KeyboardInterrupt:
        timeline.request_stop()
        stop_monitor()
    finally:
        stop_process(process)

    return_code = process.wait()
    print(
        f"recording finished with {samples_written} samples, output={output_path}",
        file=sys.stderr,
        flush=True,
    )

    if return_code not in (0, 128 + signal.SIGINT, -signal.SIGINT):
        print(
            f"monitor command exited with code {return_code}",
            file=sys.stderr,
            flush=True,
        )
        return return_code if return_code >= 0 else 1

    return 0


def parse_sample(line: str) -> Sample | None:
    match = DATA_LINE_RE.search(line)
    if match is None:
        return None

    acceleration = parse_vector(match.group("acc"))
    velocity = parse_vector(match.group("vel"))
    if acceleration is None or velocity is None:
        return None

    return Sample(
        timestamp_ms=monotonic_ms(),
        acceleration=acceleration,
        velocity=velocity,
    )


def parse_vector(raw: str) -> tuple[float, float, float] | None:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 3:
        return None

    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return None


def resolve_output_dir(base_dir: Path, raw_output_dir: str) -> Path:
    output_dir = Path(raw_output_dir).expanduser()
    if output_dir.is_absolute():
        return output_dir
    return (base_dir / output_dir).resolve()


def utc_filename_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def monotonic_ms() -> int:
    return int(time.monotonic() * 1_000)


def relative_timestamp_ms(timestamp_ms: int, start_ms: int) -> int:
    return max(0, timestamp_ms - start_ms)


def format_monotonic_ms(timestamp_ms: int) -> str:
    return f"{timestamp_ms} ms"


def slugify_filename(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", raw.strip())
    cleaned = cleaned.strip("-._")
    if not cleaned:
        raise ValueError(
            "custom name must contain at least one filename-safe character"
        )
    return cleaned


def stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return

    try:
        process.send_signal(signal.SIGINT)
        process.wait(timeout=1.5)
        return
    except (subprocess.TimeoutExpired, ProcessLookupError):
        pass

    process.terminate()
    try:
        process.wait(timeout=1.0)
        return
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=1.0)


if __name__ == "__main__":
    raise SystemExit(main())
