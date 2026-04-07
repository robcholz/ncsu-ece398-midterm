# Finn Sheng Contribution Reflection

This note is a structured summary of my work on the ECE398 midterm project based on the commit history. It is written so I can explain both what I built and how I approached the project.

## Scope of my contribution

If William covers the Python visualizer and the graphs from the PR, I should focus my part on the embedded and data-pipeline side:

- bringing up the Rust firmware and monitor workflow
- making the board reliably talk to the external IMU
- building the recording pipeline that produced the raw CSV data
- expanding the labeled dataset across the event classes
- cleaning up the toolchain and organizing outputs for handoff

## Project Context

We needed a practical end-to-end workflow for collecting IMU data from the board and turning it into usable labeled recordings for the midterm. At the start, that meant more than just writing firmware. We needed a reliable way to:

- connect to the hardware
- identify the correct sensor and bus
- stream data in a stable format
- record data on the host machine
- label and organize recordings for later analysis

Without that foundation, the visualizer and any downstream graphs would not have had dependable input data.

## What I Needed To Do

My task was to make the sensing pipeline real and repeatable. Concretely, that meant:

- building the Rust firmware for the STM32 board
- probing the IMU correctly and refusing unsafe fallback behavior
- exposing a usable USB CDC console for host-side monitoring
- creating a host workflow to flash, monitor, and record data
- collecting labeled recordings for multiple event categories
- leaving the repo in a state that teammates could use without reverse engineering my setup

## What I Did

I approached the work in stages.

### 1. I built the embedded foundation first

In the early Rust commits, I set up the embedded project structure, the build configuration, and the board firmware. The firmware uses Embassy, configures USB CDC, sets up the I2C buses, and streams sensor data out in a host-readable form.

This part of the work matters because I did not treat the board as a black box. I made the firmware explicit about which bus was being used and what sensor was actually detected. In `src/main.rs`, I added logic to:

- initialize and report all four board I2C buses
- probe WHO_AM_I values
- distinguish between the `LSM6DSV16X` and `LSM6DSO16IS`
- report which bus, pins, and address were active
- prefer the external IMU path
- refuse to silently fall back to the internal sensor when the external probe failed

More specifically, the firmware reports the board layout for `I2C1`, `I2C2`, `I2C3`, and `I2C4`, then probes the external DIL24 bus on `I2C3 (PG7/PG8)` for the `LSM6DSO16IS` at `0x6A/0x6B`. If that probe fails, it prints a clear failure message instead of quietly switching over to the internal `LSM6DSV16X` on `I2C1`.

That reflects how I think technically: I would rather fail loudly with a precise message than produce data from the wrong source and contaminate the dataset.

### 2. I made the hardware workflow usable from the host

I then built the support tooling around the firmware, including:

- `scripts/flash.sh`
- `tools/monitor`
- `scripts/monitor_record.py`
- `scripts/monitor_plot.py`

The `tools/monitor` Rust utility auto-detects the USB modem device, configures the serial port, and supports read-only monitoring. This reduced setup friction and made the board easier to use repeatedly during collection sessions.

It is a small detail, but an important one: I made the monitor usable in both an automatic and manual mode. It can scan `/dev/cu.usbmodem*`, use a product hint to choose the likely device, accept an explicit `--port`, print the chosen port, or drop into `screen` if needed. That kept the normal path simple while still leaving an escape hatch for debugging.

My reasoning here was simple: if the workflow is annoying, data collection becomes inconsistent. So I tried to remove manual steps that could cause mistakes, especially around port detection and board communication.

### 3. I added calibration and recording support so the data would be usable

The firmware does not just dump raw values. I added logic for:

- sensor configuration
- bias calibration while the board is still
- periodic sampling every 20 ms
- acceleration streaming
- velocity integration with explicit warning that it is drift-prone

The calibration path is concrete in the code: the firmware asks the user to keep the board still for about 3 seconds, averages `128` samples, subtracts that bias from acceleration, and then streams both corrected acceleration and integrated velocity.

That choice shows another part of my thinking: I tried to be honest about signal quality. I included the useful derived signal, but I also documented its limitation instead of pretending it was perfect.

On the host side, I added the Python recording script so the streamed data could be saved into CSV files with timestamps and labels. That made the pipeline reproducible instead of ad hoc. The recorder launches `cargo monitor`, parses lines shaped like `acc=[...] velocity=[...]`, writes a CSV with `timestamp_utc`, acceleration, velocity, and `label`, and supports simple live controls:

- `1` to start a labeled region
- `0` to stop a labeled region
- `q` to stop recording cleanly

That choice was intentional. I wanted labeling to happen during collection with minimal friction, so I used a binary timeline model instead of making labeling a separate manual cleanup step.

### 4. I expanded the dataset class by class

After the pipeline was working, I used it to collect and commit labeled recordings for:

- speech
- deep_breath
- cough
- throat_clear
- groan
- laugh
- sneeze

This was not one giant dump at the end. The commit history shows an incremental process where I added classes one at a time and updated the dataset documentation as the work matured.

That reflects a deliberate strategy: validate the pipeline on a smaller case first, then scale the dataset once collection and labeling are stable. The sequence in the history shows that progression clearly: I added `speech` first, then moved through `deep_breath`, `cough`, `throat_clear`, `groan`, `laugh`, and `sneeze` as the collection workflow became more dependable.

### 5. I cleaned up the project for team use

Near the end, I added toolchain requirements to the README, renamed the project cleanly, migrated the Python workflow to `uv`, and reorganized screenshot outputs. These are not the flashiest changes, but they matter in a team repo because they lower the cost for the next person trying to run the project.

There is also a design pattern in the supporting scripts that I think is worth noting. The original live plotter was deliberately lightweight: it used the Python standard library plus `tkinter` drawing on a canvas rather than depending on a heavier plotting stack. Later cleanup moved the Python environment to `uv` and made the project easier to reproduce from a fresh machine.

This is also part of how I think: finishing the technical work is not enough if the repo is still hard to understand or hard to reproduce.

## Outcome

The result of my work was an end-to-end collection pipeline that the team could build on:

- the board firmware could detect and stream the IMU data reliably
- the host machine had a repeatable flash and monitor workflow
- recordings could be captured into structured CSV files
- multiple labeled event categories were collected and committed
- the repo documented the toolchain needed to reproduce the setup
- William could then build on top of this with the Python visualizer and graph presentation

In other words, my work turned the project from a hardware idea into a usable data source. The graphs and visual analysis depended on that pipeline being stable first.

## How I think

Based on the commit sequence, the main pattern in my work is:

- I establish the system boundary first. I want to know exactly which hardware, bus, and data path are active.
- I prefer reliable failure over silent fallback. If the wrong IMU is being read, I would rather stop than generate misleading data.
- I build tooling around the core system early. Monitoring, recording, and flashing are part of the product, not side tasks.
- I iterate from narrow to broad. I got one workflow working, then scaled it to more labels and more data.
- I leave behind runnable infrastructure. README updates, toolchain notes, and script cleanup are part of the engineering result.

## Commit evidence

The main commits that support this story are:

- `f812c19` `feat: init commit`
  initial Rust firmware, USB CDC setup, monitor utility, flash script, and plotting skeleton
- `55ea57f` `refactor: used external imu`
  external IMU path, bus reporting, sensor probing, and explicit no-fallback behavior
- `eb0e732` `feat: added record py script`
  host-side recording pipeline with live label controls and CSV output
- `0f13a59` `refactor: warpped up everything`
  workflow refinement across plot and record scripts
- `07d366d` `refactor: highlighted the event period in plotting`
  clearer event interpretation during analysis by shading labeled regions
- `c2d8569` through `b721efd`
  labeled dataset expansion across speech and the other event classes
- `059b8a8` `doc: added toolchain requirements`
  reproducibility and teammate onboarding
- `3d13723` `refactor(script): use uv`
  cleaner Python environment management
- `2ad1de4` `refactor(visual): reorgaized`
  screenshot organization and repo cleanup

## Short version I can say out loud

"My main contribution was building the pipeline that made the dataset possible. I focused on the Rust and hardware side first: getting the STM32 firmware working, making sure we were reading the correct external IMU, exposing the data through USB CDC, and adding calibration and monitoring support. Then I built the host-side recording workflow and used it to collect and organize labeled recordings across all of our event classes. My teammate can cover the Python visualizer and the graphs, but the part I owned was making sure we had a reliable, repeatable source of data in the first place."
