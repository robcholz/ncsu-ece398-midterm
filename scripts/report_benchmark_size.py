#!/usr/bin/env python3
"""Build the embedded benchmark binary and report flash-size metrics.

The benchmark runtime prints model-specific sizes over USB. This script covers
the remaining deployment metric from docs/tasks.md: flash size for the final
benchmark ELF.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

DEFAULT_TARGET = "thumbv8m.main-none-eabihf"
DEFAULT_BIN = "benchmark"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Build the embedded benchmark binary and report flash size."
    )
    parser.add_argument(
        "--cwd",
        default=str(repo_root),
        help="repository root to build from (default: %(default)s)",
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help="Rust target triple (default: %(default)s)",
    )
    parser.add_argument(
        "--bin",
        default=DEFAULT_BIN,
        help="benchmark binary name (default: %(default)s)",
    )
    return parser.parse_args()


def run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )


def locate_llvm_size(cwd: Path) -> Path:
    rustc = run_command(["rustup", "which", "rustc"], cwd=cwd).stdout.strip()
    if not rustc:
        raise FileNotFoundError("rustup which rustc returned an empty path")

    rustc_path = Path(rustc).resolve()
    toolchain_root = rustc_path.parents[1]
    host = run_command(["rustc", "-vV"], cwd=cwd).stdout
    match = re.search(r"host:\s+(\S+)", host)
    if match is None:
        raise RuntimeError("unable to determine Rust host triple from rustc -vV")

    llvm_size = toolchain_root / "lib" / "rustlib" / match.group(1) / "bin" / "llvm-size"
    if not llvm_size.exists():
        raise FileNotFoundError(
            f"llvm-size not found at {llvm_size}. Install llvm-tools-preview with rustup."
        )
    return llvm_size


def parse_size_output(output: str) -> tuple[int, int, int]:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) < 2:
        raise RuntimeError(f"unexpected llvm-size output:\n{output}")

    fields = lines[-1].split()
    if len(fields) < 6:
        raise RuntimeError(f"unexpected llvm-size row:\n{lines[-1]}")

    text = int(fields[0])
    data = int(fields[1])
    bss = int(fields[2])
    return text, data, bss


def main() -> int:
    args = parse_args()
    cwd = Path(args.cwd).expanduser().resolve()

    try:
        build = run_command(
            [
                "cargo",
                "build",
                "--target",
                args.target,
                "--bin",
                args.bin,
                "--release",
            ],
            cwd=cwd,
        )
        if build.stderr:
            print(build.stderr, file=sys.stderr, end="")

        llvm_size = locate_llvm_size(cwd)
        elf_path = cwd / "target" / args.target / "release" / args.bin
        size = run_command([str(llvm_size), str(elf_path)], cwd=cwd)
        text, data, bss = parse_size_output(size.stdout)
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    flash_size = text + data

    print(f"binary={elf_path}")
    print(f"flash_size_bytes={flash_size}")
    print(f"text_bytes={text}")
    print(f"data_bytes={data}")
    print(f"bss_bytes={bss}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
