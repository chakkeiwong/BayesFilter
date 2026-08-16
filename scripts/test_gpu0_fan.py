#!/usr/bin/env python3
"""Check GPU0 fan telemetry and, optionally, exercise X-server fan control.

The default invocation is read-only.  ``nvidia-smi`` GPU indices and
``nvidia-settings`` GPU indices are not guaranteed to be the same, so the
script reports both the PCI bus identity and the X-server fan targets rather
than silently assigning a fan on a guessed target.

Manual control is deliberately opt-in::

    python scripts/test_gpu0_fan.py --set-percent 40 --confirm-manual-control

The manual probe restores automatic driver control in a ``finally`` block.
It requires an X display and an NVIDIA X configuration with Coolbits fan
control enabled; it does not change the X configuration.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from typing import Sequence


GPU_QUERY = (
    "index,uuid,name,pci.bus_id,temperature.gpu,fan.speed,utilization.gpu"
)


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _nvidia_smi_gpu(index: int) -> dict[str, str]:
    result = _run(
        [
            "nvidia-smi",
            f"--id={index}",
            f"--query-gpu={GPU_QUERY}",
            "--format=csv,noheader,nounits",
        ]
    )
    if result.returncode:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    rows = list(csv.reader([line for line in result.stdout.splitlines() if line.strip()]))
    if len(rows) != 1 or len(rows[0]) != len(GPU_QUERY.split(",")):
        raise RuntimeError(f"unexpected nvidia-smi output: {result.stdout.strip()!r}")
    return {key: value.strip() for key, value in zip(GPU_QUERY.split(","), rows[0])}


def _x_gpu_ids() -> list[int]:
    result = _run(["nvidia-settings", "-q", "gpus"])
    if result.returncode:
        raise RuntimeError(f"nvidia-settings GPU query failed: {result.stderr.strip()}")
    return sorted({int(value) for value in re.findall(r"\[gpu:(\d+)\]", result.stdout)})


def _x_value(target: str, attribute: str) -> str:
    result = _run(["nvidia-settings", "-t", "-q", f"[{target}]/{attribute}"])
    if result.returncode:
        return "unavailable"
    return result.stdout.strip()


def _print_report(gpu: dict[str, str]) -> None:
    print(
        f"GPU{gpu['index']}: {gpu['name']}\n"
        f"  UUID: {gpu['uuid']}\n"
        f"  PCI bus: {gpu['pci.bus_id']}\n"
        f"  Temperature: {gpu['temperature.gpu']} C\n"
        f"  nvidia-smi fan: {gpu['fan.speed']}%\n"
        f"  GPU utilization: {gpu['utilization.gpu']}%"
    )
    try:
        x_ids = _x_gpu_ids()
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"X fan control: unavailable ({exc})")
        return

    print("X-server targets (read-only):")
    for x_id in x_ids:
        bus = _x_value(f"gpu:{x_id}", "PCIBus")
        state = _x_value(f"gpu:{x_id}", "GPUFanControlState")
        print(f"  gpu:{x_id}: PCI bus {bus}, manual-control {state}")
    for fan_id in range(8):
        speed = _x_value(f"fan:{fan_id}", "GPUCurrentFanSpeed")
        if speed == "unavailable":
            break
        rpm = _x_value(f"fan:{fan_id}", "GPUCurrentFanSpeedRPM")
        target = _x_value(f"fan:{fan_id}", "GPUTargetFanSpeed")
        print(f"  fan:{fan_id}: current {speed}%, target {target}%, {rpm} RPM")


def _manual_probe(percent: int, x_gpu: int, x_fan: int) -> None:
    if not os.environ.get("DISPLAY"):
        raise RuntimeError("manual control requires DISPLAY to address the NVIDIA X server")
    if not 0 <= percent <= 100:
        raise ValueError("--set-percent must be between 0 and 100")
    target_gpu = f"[gpu:{x_gpu}]/GPUFanControlState"
    target_fan = f"[fan:{x_fan}]/GPUTargetFanSpeed"
    print(f"Setting {target_gpu}=1 and {target_fan}={percent} for a read-back probe")
    try:
        for assignment in (f"{target_gpu}=1", f"{target_fan}={percent}"):
            result = _run(["nvidia-settings", "-a", assignment])
            if result.returncode:
                raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        print(f"Read-back: fan { _x_value(f'fan:{x_fan}', 'GPUCurrentFanSpeed') }%")
    finally:
        restore = _run(["nvidia-settings", "-a", f"{target_gpu}=0"])
        if restore.returncode:
            print(
                "WARNING: could not restore automatic fan control: "
                f"{restore.stderr.strip()}",
                file=sys.stderr,
            )
        else:
            print("Restored automatic driver fan control")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-index", type=int, default=0, help="nvidia-smi GPU index (default: 0)")
    parser.add_argument("--set-percent", type=int, help="opt-in manual fan probe target")
    parser.add_argument("--x-gpu", type=int, help="nvidia-settings GPU target for --set-percent")
    parser.add_argument("--x-fan", type=int, help="nvidia-settings fan target for --set-percent")
    parser.add_argument(
        "--confirm-manual-control",
        action="store_true",
        help="required before changing a fan target; automatic control is restored afterward",
    )
    args = parser.parse_args(argv)
    if shutil.which("nvidia-smi") is None:
        parser.error("nvidia-smi is not installed or not on PATH")
    try:
        gpu = _nvidia_smi_gpu(args.gpu_index)
        _print_report(gpu)
        fan = int(gpu["fan.speed"])
        print("RESULT: fan telemetry is active" if fan > 0 else "RESULT: fan is at 0% (zero-RPM mode or unavailable)")
        if args.set_percent is not None:
            if not args.confirm_manual_control or args.x_gpu is None or args.x_fan is None:
                parser.error("--set-percent requires --confirm-manual-control, --x-gpu, and --x-fan")
            _manual_probe(args.set_percent, args.x_gpu, args.x_fan)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
