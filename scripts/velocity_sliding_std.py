#!/usr/bin/env python3
"""Compute centered sliding-window standard deviation for velocity columns.

The script reads one or more CSV recordings using the project's existing format
and writes new CSV files with the original columns plus:

    vel_std_x, vel_std_y, vel_std_z

The rolling output keeps the same number of rows as the source data by using a
centered window with clipped edges.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import statistics
import sys

VELOCITY_FIELDS = ("vel_x", "vel_y", "vel_z")
OUTPUT_FIELDS = ("vel_std_x", "vel_std_y", "vel_std_z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Process IMU velocity columns with a centered sliding-window "
            "standard deviation."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="CSV file(s) or directories containing CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "write processed files under this directory; when omitted, each file "
            "is written next to the source with a suffix"
        ),
    )
    parser.add_argument(
        "--suffix",
        default="_vel_std",
        help="suffix added to processed filenames when --output-dir is omitted",
    )
    return parser.parse_args()


def discover_csv_files(inputs: list[str], skip_suffix: str) -> list[Path]:
    csv_files: list[Path] = []
    for raw_input in inputs:
        path = Path(raw_input).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"input path does not exist: {path}")
        if path.is_dir():
            csv_files.extend(
                sorted(
                    candidate
                    for candidate in path.rglob("*.csv")
                    if not candidate.name.endswith(f"{skip_suffix}.csv")
                )
            )
        elif path.is_file() and path.suffix.lower() == ".csv":
            if path.name.endswith(f"{skip_suffix}.csv"):
                continue
            csv_files.append(path)
        else:
            raise ValueError(f"input is not a CSV file or directory: {path}")

    unique_files: list[Path] = []
    seen: set[Path] = set()
    for csv_file in csv_files:
        if csv_file in seen:
            continue
        seen.add(csv_file)
        unique_files.append(csv_file)

    if not unique_files:
        raise ValueError("no CSV files found in the provided inputs")
    return unique_files


def load_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path}: missing CSV header row")

        missing_fields = [
            field for field in VELOCITY_FIELDS if field not in reader.fieldnames
        ]
        if missing_fields:
            missing = ", ".join(missing_fields)
            raise ValueError(f"{csv_path}: missing required columns: {missing}")

        rows = list(reader)
        return list(reader.fieldnames), rows


def choose_window_size(sample_count: int) -> int:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")

    auto_window = 5 + min(5, sample_count // 200)
    return min(sample_count, auto_window)


def centered_window_bounds(index: int, total: int, window_size: int) -> tuple[int, int]:
    left = index - (window_size // 2)
    right = left + window_size

    if left < 0:
        right -= left
        left = 0

    if right > total:
        left -= right - total
        right = total
        left = max(0, left)

    return left, right


def rolling_std(values: list[float], window_size: int) -> list[float]:
    if not values:
        return []

    output: list[float] = []
    total = len(values)
    for index in range(total):
        left, right = centered_window_bounds(index, total, window_size)
        window = values[left:right]
        output.append(statistics.pstdev(window) if len(window) > 1 else 0.0)
    return output


def resolve_output_path(
    source_path: Path,
    output_dir: Path | None,
    suffix: str,
    common_root: Path,
) -> Path:
    if output_dir is None:
        return source_path.with_name(f"{source_path.stem}{suffix}{source_path.suffix}")

    relative_path = source_path.relative_to(common_root)
    return output_dir / relative_path


def write_processed_csv(
    source_path: Path,
    output_path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
    window_size: int,
    std_x: list[float],
    std_y: list[float],
    std_z: list[float],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    new_fieldnames = list(fieldnames)
    for field in OUTPUT_FIELDS:
        if field not in new_fieldnames:
            new_fieldnames.append(field)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=new_fieldnames)
        writer.writeheader()
        for row, vel_std_x, vel_std_y, vel_std_z in zip(
            rows, std_x, std_y, std_z, strict=True
        ):
            output_row = dict(row)
            output_row["vel_std_x"] = f"{vel_std_x:.6f}"
            output_row["vel_std_y"] = f"{vel_std_y:.6f}"
            output_row["vel_std_z"] = f"{vel_std_z:.6f}"
            writer.writerow(output_row)

    print(
        f"processed {source_path} -> {output_path} "
        f"(rows={len(rows)}, window={window_size})"
    )


def process_file(
    source_path: Path,
    output_path: Path,
) -> None:
    fieldnames, rows = load_rows(source_path)
    if not rows:
        raise ValueError(f"{source_path}: CSV has no data rows")

    window_size = choose_window_size(len(rows))

    velocity_columns: list[list[float]] = [[], [], []]
    for row_number, row in enumerate(rows, start=2):
        for axis_index, field in enumerate(VELOCITY_FIELDS):
            raw_value = row.get(field, "")
            try:
                velocity_columns[axis_index].append(float(raw_value))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{source_path}: row {row_number}: invalid {field} value {raw_value!r}"
                ) from exc

    std_columns = [
        rolling_std(values=axis_values, window_size=window_size)
        for axis_values in velocity_columns
    ]

    write_processed_csv(
        source_path=source_path,
        output_path=output_path,
        fieldnames=fieldnames,
        rows=rows,
        window_size=window_size,
        std_x=std_columns[0],
        std_y=std_columns[1],
        std_z=std_columns[2],
    )


def main() -> int:
    args = parse_args()

    try:
        csv_files = discover_csv_files(args.inputs, skip_suffix=args.suffix)
        common_root = os.path.commonpath([str(path.parent) for path in csv_files])
        common_root_path = Path(common_root)

        for csv_file in csv_files:
            output_path = resolve_output_path(
                source_path=csv_file,
                output_dir=args.output_dir.resolve() if args.output_dir else None,
                suffix=args.suffix,
                common_root=common_root_path,
            )
            process_file(csv_file, output_path)
    except (FileNotFoundError, OSError, ValueError, csv.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
