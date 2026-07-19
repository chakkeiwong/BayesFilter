#!/usr/bin/env python3
"""Wait for the assigned GPU to clear, then exec the reviewed Phase 6 canary."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(
    "docs/benchmarks/"
    "run_ssl_lstm_neutra_phase6_transformed_hmc_tuning_2026_07_16.py"
)
HMC_SOURCE = Path("bayesfilter/inference/hmc.py")


class QueueError(RuntimeError):
    """Raised when queued execution cannot preserve its reviewed boundary."""


def sha256(path: Path) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def pid_command(pid: int, *, proc_root: Path = Path("/proc")) -> str | None:
    path = proc_root / str(pid) / "cmdline"
    try:
        payload = path.read_bytes()
    except FileNotFoundError:
        return None
    return payload.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()


def gpu_compute_rows() -> tuple[tuple[str, str], ...]:
    output = subprocess.run(
        (
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    rows = []
    for line in output.splitlines():
        fields = tuple(item.strip() for item in line.split(","))
        if len(fields) == 2 and fields[0]:
            rows.append((fields[0], fields[1]))
    return tuple(rows)


def gpu_is_busy(gpu_uuid: str, rows: tuple[tuple[str, str], ...]) -> bool:
    return any(row_uuid == gpu_uuid for row_uuid, _pid in rows)


def wait_for_gpu(
    *,
    wait_pid: int,
    wait_command_substring: str,
    gpu_uuid: str,
    poll_seconds: float,
    max_wait_seconds: float,
) -> None:
    started = time.monotonic()
    initial_command = pid_command(wait_pid)
    if initial_command is not None and wait_command_substring not in initial_command:
        raise QueueError("wait PID no longer belongs to the declared other-lane command")
    clear_observations = 0
    while True:
        elapsed = time.monotonic() - started
        if elapsed > max_wait_seconds:
            raise QueueError("queued Phase 6 canary exhausted its wait cap")
        current_command = pid_command(wait_pid)
        if current_command is not None:
            if wait_command_substring not in current_command:
                raise QueueError("wait PID was reused by an unrelated process")
            clear_observations = 0
        elif gpu_is_busy(gpu_uuid, gpu_compute_rows()):
            clear_observations = 0
        else:
            clear_observations += 1
            if clear_observations >= 2:
                return
        time.sleep(poll_seconds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--wait-command-substring", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--max-wait-seconds", type=float, required=True)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--hmc-sha256", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-cap-seconds", type=float, required=True)
    parser.add_argument("--child-timeout-seconds", type=int, required=True)
    args = parser.parse_args(argv)
    if args.wait_pid <= 0:
        raise QueueError("wait PID must be positive")
    if args.poll_seconds <= 0.0 or args.max_wait_seconds <= 0.0:
        raise QueueError("queue timing limits must be positive")
    if args.output.is_absolute():
        raise QueueError("output must be repository-relative")
    if (ROOT / args.output).exists():
        raise QueueError("Phase 6 canary output already exists")
    wait_for_gpu(
        wait_pid=args.wait_pid,
        wait_command_substring=args.wait_command_substring,
        gpu_uuid=args.gpu_uuid,
        poll_seconds=args.poll_seconds,
        max_wait_seconds=args.max_wait_seconds,
    )
    if sha256(RUNNER) != args.runner_sha256:
        raise QueueError("reviewed Phase 6 runner hash drifted while queued")
    if sha256(HMC_SOURCE) != args.hmc_sha256:
        raise QueueError("reviewed shared HMC source hash drifted while queued")
    if sha256(args.plan) != args.plan_sha256:
        raise QueueError("reviewed Phase 6 plan hash drifted while queued")
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": "/tmp/bayesfilter-phase6-pyc",
            "CUDA_CACHE_PATH": "/tmp/bayesfilter-phase6-cuda",
        }
    )
    command = (
        "timeout",
        str(args.child_timeout_seconds),
        "/home/ubuntu/anaconda3/envs/tfgpu/bin/python",
        RUNNER.as_posix(),
        "--stage",
        "canary",
        "--output",
        args.output.as_posix(),
        "--wall-cap-seconds",
        str(args.wall_cap_seconds),
    )
    os.chdir(ROOT)
    os.execvpe(command[0], command, environment)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
