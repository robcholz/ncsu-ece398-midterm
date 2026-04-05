#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <firmware-elf>" >&2
  exit 2
fi

ELF="$1"
BIN="${ELF}.bin"

if ! command -v rustup >/dev/null 2>&1; then
  echo "rustup is required to locate llvm-objcopy" >&2
  exit 1
fi

if ! command -v dfu-util >/dev/null 2>&1; then
  echo "dfu-util is required to flash this STM32 board over USB DFU" >&2
  exit 1
fi

SYSROOT="$(rustc --print sysroot)"
HOST="$(rustc -vV | sed -n 's/^host: //p')"
OBJCOPY="${SYSROOT}/lib/rustlib/${HOST}/bin/llvm-objcopy"

if [ ! -x "$OBJCOPY" ]; then
  echo "missing llvm-objcopy; install it with: rustup component add llvm-tools-preview" >&2
  exit 1
fi

"$OBJCOPY" -O binary "$ELF" "$BIN"

LOG_FILE="$(mktemp)"
if dfu-util -a 0 -s 0x08000000:leave -D "$BIN" >"$LOG_FILE" 2>&1; then
  cat "$LOG_FILE"
  rm -f "$LOG_FILE"
  exit 0
else
  STATUS="$?"
fi
cat "$LOG_FILE"

if grep -q "File downloaded successfully" "$LOG_FILE" \
  && grep -q "Error during download get_status" "$LOG_FILE"; then
  echo "dfu-util leave raced the device reboot; treating flash as successful" >&2
  rm -f "$LOG_FILE"
  exit 0
fi

rm -f "$LOG_FILE"
exit "$STATUS"
