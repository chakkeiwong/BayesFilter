#!/usr/bin/env python
"""Compare pinned CPU/XLA architectures for the analytical Kalman QR score."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import selectors
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = Path(__file__).resolve()
PYTHON = Path("/home/ubuntu/anaconda3/envs/tfgpu/bin/python")
HWLOC_BIND = Path("/home/ubuntu/anaconda3/envs/tfgpu/bin/hwloc-bind")
PLAN = "docs/plans/bayesfilter-kalman-qr-cpu-throughput-comparison-plan-2026-07-14.md"
RESULT = "docs/plans/bayesfilter-kalman-qr-cpu-throughput-comparison-result-2026-07-14.md"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "docs/benchmarks/kalman_qr_cpu_throughput_comparison_2026-07-14"
SCHEMA = "bayesfilter.kalman_qr.cpu_throughput_comparison.v1"
BLOCK_SCHEMA = "bayesfilter.kalman_qr.cpu_throughput_block.v1"
WORKER_SCHEMA = "bayesfilter.kalman_qr.cpu_throughput_worker.v1"
SOURCE_PATHS = (
    "bayesfilter/linear/kalman_qr_tf.py",
    "bayesfilter/linear/kalman_qr_derivatives_tf.py",
    "bayesfilter/linear/qr_factor_tf.py",
    "scripts/benchmark_kalman_qr_parameter_count_scaling.py",
    "scripts/kalman_qr_benchmark_contract.py",
    "docs/benchmarks/run_kalman_qr_cpu_throughput_comparison_2026_07_14.py",
    PLAN,
)
CANONICAL_ROWS = tuple(range(16))
CORE_BUDGETS = (1, 2, 4, 8, 16)
PHYSICAL_CPU_POOL = tuple(range(16, 32))
EXCLUDED_SMT_SIBLINGS = tuple(range(144, 160))
FLOAT32_PARITY = {
    "value_rtol": 2.0e-4,
    "value_atol": 2.0e-4,
    "score_rtol": 2.0e-4,
    "score_atol": 2.0e-4,
}
RANDOM_SEED = 20260714
ROUNDS = 5
WARM_ROUNDS = 2
MAX_RSS_BYTES = 32 * 1024**3
READY_TIMEOUT_SECONDS = 900
ROUND_TIMEOUT_SECONDS = 300
BLOCK_TIMEOUT_SECONDS = 1800
RESOURCE_WAIT_SECONDS = 7200
RESOURCE_POLL_SECONDS = 30
MAX_PRELAUNCH_TARGET_BUSY_FRACTION = 0.10
MAX_PRELAUNCH_LOAD_ONE_MINUTE = 16.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest() -> dict[str, Any]:
    files = []
    for relative in SOURCE_PATHS:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"missing source path: {relative}")
        files.append({"path": relative, "sha256": _sha256(path)})
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"files": files, "fingerprint": hashlib.sha256(encoded).hexdigest()}


def partition_rows(core_budget: int) -> list[list[int]]:
    if core_budget not in CORE_BUDGETS:
        raise ValueError(f"unsupported core budget: {core_budget}")
    partitions = [list(CANONICAL_ROWS[index::core_budget]) for index in range(core_budget)]
    flattened = [row for partition in partitions for row in partition]
    if sorted(flattened) != list(CANONICAL_ROWS) or len(flattened) != len(set(flattened)):
        raise RuntimeError("row partition does not cover canonical rows exactly once")
    return partitions


def cpu_list(core_budget: int) -> tuple[int, ...]:
    if core_budget not in CORE_BUDGETS:
        raise ValueError(f"unsupported core budget: {core_budget}")
    return PHYSICAL_CPU_POOL[:core_budget]


def parse_cpu_list(value: str) -> set[int]:
    parsed: set[int] = set()
    for part in value.strip().split(","):
        if not part:
            continue
        if "-" in part:
            first, last = (int(item) for item in part.split("-", 1))
            parsed.update(range(first, last + 1))
        else:
            parsed.add(int(part))
    return parsed


def topology_contract() -> dict[str, Any]:
    completed = subprocess.run(
        ["lscpu", "-e=CPU,NODE,SOCKET,CORE,ONLINE"],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    rows = []
    for line in completed.stdout.splitlines()[1:]:
        fields = line.split()
        if len(fields) != 5:
            continue
        rows.append(
            {
                "cpu": int(fields[0]),
                "node": int(fields[1]),
                "socket": int(fields[2]),
                "core": int(fields[3]),
                "online": fields[4].lower() == "yes",
            }
        )
    by_cpu = {row["cpu"]: row for row in rows}
    selected = [by_cpu.get(cpu) for cpu in PHYSICAL_CPU_POOL]
    siblings = [by_cpu.get(cpu) for cpu in EXCLUDED_SMT_SIBLINGS]
    valid = (
        completed.returncode == 0
        and all(row is not None for row in selected)
        and all(row["online"] and row["node"] == 0 and row["socket"] == 0 for row in selected if row)
        and len({row["core"] for row in selected if row}) == 16
        and all(row is not None for row in siblings)
        and all(
            sibling["node"] == 0
            and sibling["socket"] == 0
            and sibling["core"] == primary["core"]
            for primary, sibling in zip(selected, siblings, strict=True)
            if primary is not None and sibling is not None
        )
        and not set(PHYSICAL_CPU_POOL).intersection(EXCLUDED_SMT_SIBLINGS)
    )
    return {
        "returncode": completed.returncode,
        "selected": selected,
        "excluded_smt_siblings": siblings,
        "valid": valid,
    }


def _anonymous_numa_pages() -> dict[str, Any]:
    totals: dict[int, int] = {}
    try:
        lines = Path("/proc/self/numa_maps").read_text(encoding="utf-8").splitlines()
    except OSError:
        return {"pages_by_node": {}, "node0_fraction": None, "valid": False}
    for line in lines:
        fields = line.split()
        if not any(field.startswith("anon=") for field in fields):
            continue
        for field in fields:
            if field.startswith("N") and "=" in field:
                node, raw_pages = field.split("=", 1)
                if node[1:].isdigit() and raw_pages.isdigit():
                    totals[int(node[1:])] = totals.get(int(node[1:]), 0) + int(raw_pages)
    total = sum(totals.values())
    fraction = totals.get(0, 0) / total if total else None
    return {
        "pages_by_node": {str(key): value for key, value in sorted(totals.items())},
        "node0_fraction": fraction,
        "valid": fraction is not None and fraction >= 0.95,
    }


def _task_affinities(pid: int) -> dict[str, Any]:
    rows: dict[str, list[int]] = {}
    for status_path in Path(f"/proc/{pid}/task").glob("*/status"):
        try:
            lines = status_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        match = next((line for line in lines if line.startswith("Cpus_allowed_list:")), None)
        if match is not None:
            rows[status_path.parent.name] = sorted(parse_cpu_list(match.split(":", 1)[1]))
    return {"task_cpu_lists": rows}


def validate_worker_placement(worker: Mapping[str, Any], expected: set[int]) -> bool:
    affinity = set(worker.get("affinity", []))
    tasks = worker.get("task_affinities", {}).get("task_cpu_lists", {})
    numa = worker.get("numa", {})
    return (
        affinity == expected
        and bool(tasks)
        and all(set(value) == expected for value in tasks.values())
        and numa.get("valid") is True
    )


def _rss_bytes(pid: int) -> int | None:
    try:
        fields = Path(f"/proc/{pid}/statm").read_text(encoding="utf-8").split()
        return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


def _process_cpu_seconds(pid: int) -> float | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        ticks = int(fields[13]) + int(fields[14])
        return ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return None


def _read_first(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def resource_snapshot(cpus: Sequence[int]) -> dict[str, Any]:
    frequencies = {}
    governors = {}
    for cpu in cpus:
        root = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq")
        frequencies[str(cpu)] = _read_first(root / "scaling_cur_freq")
        governors[str(cpu)] = _read_first(root / "scaling_governor")
    meminfo = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            name, raw = line.split(":", 1)
            if name in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                meminfo[name] = raw.strip()
    except OSError:
        pass
    completed = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,psr=,stat=,pcpu=,nlwp=,args="],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    process_rows = []
    for line in completed.stdout.splitlines():
        fields = line.strip().split(None, 6)
        if len(fields) != 7:
            continue
        try:
            last_cpu = int(fields[2])
            cpu_percent = float(fields[4])
        except ValueError:
            continue
        if last_cpu in cpus or cpu_percent >= 50.0:
            process_rows.append(
                {
                    "pid": int(fields[0]),
                    "ppid": int(fields[1]),
                    "last_cpu": last_cpu,
                    "state": fields[3],
                    "cpu_percent": cpu_percent,
                    "threads": int(fields[5]),
                    "command": fields[6],
                }
            )
    temperatures = {}
    for path in sorted(Path("/sys/class/hwmon").glob("hwmon*/temp*_input")):
        value = _read_first(path)
        if value is not None:
            temperatures[str(path)] = value
    return {
        "observed_utc": _utc_now(),
        "cpus": list(cpus),
        "load_average": list(os.getloadavg()),
        "frequency_khz": frequencies,
        "governor": governors,
        "temperatures_millidegree_c": temperatures,
        "memory": meminfo,
        "process_query_returncode": completed.returncode,
        "relevant_processes": process_rows,
    }


def _cpu_stat_snapshot(cpus: Sequence[int]) -> dict[int, tuple[int, int]]:
    selected = set(cpus)
    result: dict[int, tuple[int, int]] = {}
    try:
        lines = Path("/proc/stat").read_text(encoding="utf-8").splitlines()
    except OSError:
        return result
    for line in lines:
        fields = line.split()
        if not fields or not fields[0].startswith("cpu") or not fields[0][3:].isdigit():
            continue
        cpu = int(fields[0][3:])
        if cpu not in selected:
            continue
        values = [int(value) for value in fields[1:]]
        idle = sum(values[index] for index in (3, 4) if index < len(values))
        result[cpu] = (sum(values), idle)
    return result


def target_cpu_contamination_seconds(
    before: Mapping[int, tuple[int, int]],
    after: Mapping[int, tuple[int, int]],
    owned_cpu_seconds: float,
    *,
    clock_ticks: int | None = None,
) -> float | None:
    if set(before) != set(after) or not before:
        return None
    ticks = clock_ticks or os.sysconf("SC_CLK_TCK")
    busy_ticks = 0
    for cpu in before:
        total_before, idle_before = before[cpu]
        total_after, idle_after = after[cpu]
        busy_ticks += (total_after - total_before) - (idle_after - idle_before)
    return max(0.0, busy_ticks / ticks - owned_cpu_seconds)


def cpu_busy_fractions(
    before: Mapping[int, tuple[int, int]],
    after: Mapping[int, tuple[int, int]],
) -> dict[int, float] | None:
    if set(before) != set(after) or not before:
        return None
    result = {}
    for cpu in before:
        total_delta = after[cpu][0] - before[cpu][0]
        idle_delta = after[cpu][1] - before[cpu][1]
        if total_delta <= 0:
            return None
        result[cpu] = max(0.0, min(1.0, (total_delta - idle_delta) / total_delta))
    return result


def wait_for_target_cpus(
    cpus: Sequence[int],
    *,
    wait_seconds: int = RESOURCE_WAIT_SECONDS,
    poll_seconds: int = RESOURCE_POLL_SECONDS,
    sample_seconds: float = 2.0,
) -> dict[str, Any]:
    started = time.monotonic()
    last = None
    while True:
        before = _cpu_stat_snapshot(cpus)
        time.sleep(sample_seconds)
        after = _cpu_stat_snapshot(cpus)
        busy = cpu_busy_fractions(before, after)
        load = list(os.getloadavg())
        last = {
            "observed_utc": _utc_now(),
            "cpus": list(cpus),
            "sample_seconds": sample_seconds,
            "busy_fractions": None if busy is None else {str(key): value for key, value in busy.items()},
            "load_average": load,
        }
        if (
            busy is not None
            and max(busy.values(), default=1.0) < MAX_PRELAUNCH_TARGET_BUSY_FRACTION
            and load[0] <= MAX_PRELAUNCH_LOAD_ONE_MINUTE
        ):
            return last
        if time.monotonic() - started >= wait_seconds:
            raise TimeoutError(f"target CPUs did not become idle: {last}")
        time.sleep(poll_seconds)


def parity_summary(
    candidate: Mapping[int, tuple[Sequence[float], Sequence[float]]],
    reference: Mapping[int, tuple[Sequence[float], Sequence[float]]],
) -> dict[str, Any]:
    if set(candidate) != set(CANONICAL_ROWS) or set(reference) != set(CANONICAL_ROWS):
        return {"passed": False, "reason": "row_identity_mismatch"}
    max_value = 0.0
    max_score = 0.0
    allowed_value = 0.0
    allowed_score = 0.0
    finite = True
    shape = True
    for row_id in CANONICAL_ROWS:
        candidate_value, candidate_score = candidate[row_id]
        reference_value, reference_score = reference[row_id]
        shape = shape and len(candidate_value) == len(reference_value) == 1
        shape = shape and len(candidate_score) == len(reference_score)
        for left, right in zip(candidate_value, reference_value, strict=False):
            finite = finite and math.isfinite(float(left)) and math.isfinite(float(right))
            max_value = max(max_value, abs(float(left) - float(right)))
            allowed_value = max(
                allowed_value,
                FLOAT32_PARITY["value_atol"] + FLOAT32_PARITY["value_rtol"] * abs(float(right)),
            )
        for left, right in zip(candidate_score, reference_score, strict=False):
            finite = finite and math.isfinite(float(left)) and math.isfinite(float(right))
            max_score = max(max_score, abs(float(left) - float(right)))
            allowed_score = max(
                allowed_score,
                FLOAT32_PARITY["score_atol"] + FLOAT32_PARITY["score_rtol"] * abs(float(right)),
            )
    return {
        "passed": finite and shape and max_value <= allowed_value and max_score <= allowed_score,
        "finite": finite,
        "shape": shape,
        "max_value_residual": max_value,
        "max_score_residual": max_score,
        "max_allowed_value_residual": allowed_value,
        "max_allowed_score_residual": allowed_score,
        "tolerances": dict(FLOAT32_PARITY),
    }


def paired_statistics(
    candidate: Sequence[float],
    comparator: Sequence[float],
    *,
    seed: int = RANDOM_SEED,
    resamples: int = 10_000,
) -> dict[str, Any]:
    if len(candidate) != len(comparator) or not candidate or any(value <= 0 for value in (*candidate, *comparator)):
        raise ValueError("paired positive equal-length timings required")
    logs = [math.log(left / right) for left, right in zip(candidate, comparator, strict=True)]
    observed = statistics.fmean(logs)
    rng = random.Random(seed)
    boot = []
    for _ in range(resamples):
        boot.append(statistics.fmean(logs[rng.randrange(len(logs))] for _ in logs))
    boot.sort()
    lower = boot[int(0.025 * resamples)]
    upper = boot[min(resamples - 1, int(0.975 * resamples))]
    non_ties = [value for value in logs if value != 0.0]
    positives = sum(value > 0 for value in non_ties)
    n = len(non_ties)
    if n:
        tail = sum(math.comb(n, index) for index in range(0, min(positives, n - positives) + 1)) / (2**n)
        sign_p = min(1.0, 2.0 * tail)
    else:
        sign_p = 1.0
    return {
        "paired_block_count": len(logs),
        "ratios": [math.exp(value) for value in logs],
        "geometric_mean_ratio": math.exp(observed),
        "bootstrap_95_interval": [math.exp(lower), math.exp(upper)],
        "bootstrap_resamples": resamples,
        "seed": seed,
        "sign_test_two_sided_p": sign_p,
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(indexed)
    running = 0.0
    count = len(indexed)
    for rank, (index, value) in enumerate(indexed):
        running = max(running, min(1.0, (count - rank) * float(value)))
        adjusted[index] = running
    return adjusted


def _wait_until(start_ns: int) -> None:
    while True:
        remaining = start_ns - time.monotonic_ns()
        if remaining <= 0:
            return
        if remaining > 2_000_000:
            time.sleep((remaining - 1_000_000) / 1.0e9)


def _worker_main(args: argparse.Namespace) -> int:
    import tensorflow as tf

    sys.path.insert(0, str(REPO_ROOT))
    from scripts.benchmark_kalman_qr_parameter_count_scaling import (
        _make_parameter_cloud,
        _synchronize_outputs,
        build_batch_native_analytic_fn,
        build_batch_native_autodiff_fn,
        make_fixture,
    )

    started = time.perf_counter()
    tf.config.threading.set_intra_op_parallelism_threads(args.intra)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    fixture = make_fixture(args.dimension, args.parameter_count, args.timesteps, dtype=tf.float32)
    cloud = _make_parameter_cloud(fixture)
    row_ids = [int(value) for value in args.row_ids.split(",") if value]
    if args.worker_mode == "batch":
        parameters = tf.gather(cloud, row_ids, axis=0)
        builder = build_batch_native_autodiff_fn if args.method == "autodiff" else build_batch_native_analytic_fn
        concrete = builder(fixture, batch_size=len(row_ids), jit_compile=True).get_concrete_function(parameters)

        def invoke():
            return concrete(parameters)

    else:
        concrete = build_batch_native_analytic_fn(fixture, batch_size=1, jit_compile=True).get_concrete_function(cloud[:1])

        def invoke():
            outputs = []
            for row_id in row_ids:
                outputs.append(concrete(cloud[row_id : row_id + 1]))
            return (
                tf.concat([value for value, _score in outputs], axis=0),
                tf.concat([score for _value, score in outputs], axis=0),
            )

    first_started = time.perf_counter()
    outputs = invoke()
    _synchronize_outputs(outputs)
    first_seconds = time.perf_counter() - first_started
    warm_seconds = []
    for _ in range(WARM_ROUNDS):
        warm_started = time.perf_counter()
        outputs = invoke()
        _synchronize_outputs(outputs)
        warm_seconds.append(time.perf_counter() - warm_started)
    ready = {
        "schema": WORKER_SCHEMA,
        "event": "ready",
        "pid": os.getpid(),
        "row_ids": row_ids,
        "mode": args.worker_mode,
        "method": args.method,
        "affinity": sorted(os.sched_getaffinity(0)),
        "task_affinities": _task_affinities(os.getpid()),
        "numa": _anonymous_numa_pages(),
        "rss_bytes": _rss_bytes(os.getpid()),
        "cold_time_to_ready_seconds": time.perf_counter() - started,
        "first_executable_call_seconds": first_seconds,
        "warm_seconds": warm_seconds,
    }
    print(json.dumps(ready, allow_nan=False), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        command = request.get("command")
        if command == "stop":
            print(json.dumps({"event": "stopped", "pid": os.getpid()}), flush=True)
            return 0
        if command == "run":
            _wait_until(int(request["start_ns"]))
            run_started = time.perf_counter_ns()
            outputs = invoke()
            _synchronize_outputs(outputs)
            elapsed = (time.perf_counter_ns() - run_started) / 1.0e9
            print(
                json.dumps(
                    {
                        "event": "done",
                        "round": request["round"],
                        "pid": os.getpid(),
                        "kernel_seconds": elapsed,
                        "rss_bytes": _rss_bytes(os.getpid()),
                    },
                    allow_nan=False,
                ),
                flush=True,
            )
            continue
        if command == "output":
            outputs = invoke()
            _synchronize_outputs(outputs)
            value, score = outputs
            print(
                json.dumps(
                    {
                        "event": "output",
                        "pid": os.getpid(),
                        "row_ids": row_ids,
                        "value": value.numpy().tolist(),
                        "score": score.numpy().tolist(),
                    },
                    allow_nan=False,
                ),
                flush=True,
            )
            continue
        raise RuntimeError(f"unknown worker command: {command}")
    return 0


def _read_messages(processes: Sequence[subprocess.Popen[str]], timeout: float) -> list[dict[str, Any]]:
    selector = selectors.DefaultSelector()
    by_fd = {}
    for process in processes:
        if process.stdout is None:
            raise RuntimeError("worker stdout unavailable")
        selector.register(process.stdout, selectors.EVENT_READ)
        by_fd[process.stdout.fileno()] = process
    deadline = time.monotonic() + timeout
    messages = []
    pending = set(processes)
    try:
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("worker message timeout")
            events = selector.select(remaining)
            if not events:
                raise TimeoutError("worker message timeout")
            for key, _mask in events:
                process = by_fd[key.fileobj.fileno()]
                line = key.fileobj.readline()
                if not line:
                    raise RuntimeError(f"worker exited before response: {process.returncode}")
                messages.append(json.loads(line))
                pending.discard(process)
    finally:
        selector.close()
    return messages


def _send(processes: Sequence[subprocess.Popen[str]], payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(dict(payload), allow_nan=False) + "\n"
    for process in processes:
        if process.stdin is None:
            raise RuntimeError("worker stdin unavailable")
        process.stdin.write(encoded)
        process.stdin.flush()


def terminate_processes(processes: Sequence[subprocess.Popen[str]]) -> None:
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 5.0
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    for process in processes:
        if process.poll() is None:
            process.wait(timeout=5)


def _worker_command(
    *,
    cpus: Sequence[int],
    intra: int,
    mode: str,
    method: str,
    row_ids: Sequence[int],
    dimension: int,
    parameter_count: int,
    timesteps: int,
) -> list[str]:
    return [
        str(HWLOC_BIND),
        "--membind",
        "node:0",
        "--",
        "taskset",
        "-c",
        ",".join(str(value) for value in cpus),
        str(PYTHON),
        str(SCRIPT_PATH),
        "--worker",
        "--worker-mode",
        mode,
        "--method",
        method,
        "--row-ids",
        ",".join(str(value) for value in row_ids),
        "--intra",
        str(intra),
        "--dimension",
        str(dimension),
        "--parameter-count",
        str(parameter_count),
        "--timesteps",
        str(timesteps),
    ]


def worker_specs(
    architecture: str,
    core_budget: int,
) -> list[tuple[tuple[int, ...], int, Sequence[int], str]]:
    cpus = cpu_list(core_budget)
    if architecture == "batch_native":
        return [(cpus, core_budget, CANONICAL_ROWS, "batch")]
    if architecture == "sharded":
        return [
            ((cpu,), 1, rows, "sharded")
            for cpu, rows in zip(cpus, partition_rows(core_budget), strict=True)
        ]
    raise ValueError(f"unknown architecture: {architecture}")


def _spawn_workers(
    *,
    architecture: str,
    core_budget: int,
    dimension: int,
    parameter_count: int,
    timesteps: int,
    log_dir: Path,
    method: str = "analytical",
) -> tuple[list[subprocess.Popen[str]], list[Any], dict[int, dict[str, Any]]]:
    specs = worker_specs(architecture, core_budget)
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "-1",
            "TF_FORCE_GPU_ALLOW_GROWTH": "true",
            "TF_NUM_INTEROP_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    processes = []
    logs = []
    expected: dict[int, dict[str, Any]] = {}
    for index, (cpus, intra, rows, mode) in enumerate(specs):
        env = environment.copy()
        env["OMP_NUM_THREADS"] = str(intra)
        env["TF_NUM_INTRAOP_THREADS"] = str(intra)
        path = log_dir / f"worker-{index}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            _worker_command(
                cpus=cpus,
                intra=intra,
                mode=mode,
                method=method,
                row_ids=rows,
                dimension=dimension,
                parameter_count=parameter_count,
                timesteps=timesteps,
            ),
            cwd=REPO_ROOT,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=handle,
            text=True,
            start_new_session=True,
            bufsize=1,
        )
        processes.append(process)
        logs.append(handle)
        expected[process.pid] = {
            "cpus": set(cpus),
            "row_ids": list(rows),
            "mode": mode,
        }
    return processes, logs, expected


def _collect_outputs(messages: Sequence[Mapping[str, Any]]) -> dict[int, tuple[list[float], list[float]]]:
    rows: dict[int, tuple[list[float], list[float]]] = {}
    for message in messages:
        row_ids = message.get("row_ids", [])
        values = message.get("value", [])
        scores = message.get("score", [])
        if not (len(row_ids) == len(values) == len(scores)):
            raise RuntimeError("worker output row shape mismatch")
        for row_id, value, score in zip(row_ids, values, scores, strict=True):
            if int(row_id) in rows:
                raise RuntimeError("duplicate worker output row")
            rows[int(row_id)] = ([float(value)], [float(item) for item in score])
    if set(rows) != set(CANONICAL_ROWS):
        raise RuntimeError("worker outputs do not cover canonical rows")
    return rows


def run_arm(
    *,
    architecture: str,
    core_budget: int,
    dimension: int,
    parameter_count: int,
    timesteps: int,
    block_dir: Path,
    method: str = "analytical",
    measured_rounds: int = ROUNDS,
) -> dict[str, Any]:
    started_utc = _utc_now()
    started = time.monotonic()
    processes: list[subprocess.Popen[str]] = []
    logs: list[Any] = []
    deadline = time.monotonic() + BLOCK_TIMEOUT_SECONDS

    def remaining(limit: float) -> float:
        available = deadline - time.monotonic()
        if available <= 0:
            raise TimeoutError("whole-block timeout")
        return min(limit, available)

    try:
        admission = wait_for_target_cpus(cpu_list(core_budget))
        resource_before = resource_snapshot(cpu_list(core_budget))
        processes, logs, expected = _spawn_workers(
            architecture=architecture,
            core_budget=core_budget,
            dimension=dimension,
            parameter_count=parameter_count,
            timesteps=timesteps,
            log_dir=block_dir / "logs",
            method=method,
        )
        ready = _read_messages(processes, remaining(READY_TIMEOUT_SECONDS))
        ready_by_pid = {int(row.get("pid", -1)): row for row in ready}
        identity_checks = []
        placement = []
        for pid, contract in expected.items():
            row = ready_by_pid.get(pid)
            identity_checks.append(
                row is not None
                and row.get("schema") == WORKER_SCHEMA
                and row.get("event") == "ready"
                and row.get("row_ids") == contract["row_ids"]
                and row.get("mode") == contract["mode"]
                and row.get("method") == method
            )
            placement.append(
                row is not None
                and validate_worker_placement(row, contract["cpus"])
            )
        if not all(identity_checks):
            raise RuntimeError("worker identity or row assignment invalid")
        if not all(placement):
            raise RuntimeError("worker affinity or NUMA placement invalid")
        _send(processes, {"command": "output"})
        output_messages = _read_messages(processes, remaining(ROUND_TIMEOUT_SECONDS))
        outputs = _collect_outputs(output_messages)
        rounds = []
        target_cpus = cpu_list(core_budget)
        for round_index in range(measured_rounds):
            before_cpu = _cpu_stat_snapshot(target_cpus)
            before_owned = {process.pid: _process_cpu_seconds(process.pid) for process in processes}
            start_ns = time.monotonic_ns() + 50_000_000
            _send(processes, {"command": "run", "round": round_index, "start_ns": start_ns})
            messages = _read_messages(processes, remaining(ROUND_TIMEOUT_SECONDS))
            completed_ns = time.monotonic_ns()
            after_cpu = _cpu_stat_snapshot(target_cpus)
            after_owned = {process.pid: _process_cpu_seconds(process.pid) for process in processes}
            owned = sum(
                max(0.0, float(after_owned[pid]) - float(before_owned[pid]))
                for pid in before_owned
                if before_owned[pid] is not None and after_owned[pid] is not None
            )
            contamination = target_cpu_contamination_seconds(before_cpu, after_cpu, owned)
            makespan = max(0.0, (completed_ns - start_ns) / 1.0e9)
            rss_values = [value for value in (_rss_bytes(process.pid) for process in processes) if value is not None]
            aggregate_rss = sum(rss_values)
            if aggregate_rss > MAX_RSS_BYTES:
                raise MemoryError("aggregate worker RSS exceeded 32 GiB")
            threshold = max(0.25, 0.02 * core_budget * makespan)
            rounds.append(
                {
                    "round": round_index,
                    "makespan_seconds": makespan,
                    "worker_kernel_seconds": [float(row["kernel_seconds"]) for row in messages],
                    "owned_cpu_seconds": owned,
                    "unattributed_target_cpu_seconds": contamination,
                    "contamination_threshold_seconds": threshold,
                    "contaminated": contamination is None or contamination > threshold,
                    "aggregate_rss_bytes": aggregate_rss,
                }
            )
        _send(processes, {"command": "stop"})
        _read_messages(processes, remaining(30))
        for process in processes:
            process.wait(timeout=30)
        ready_rss = [int(row["rss_bytes"]) for row in ready if row.get("rss_bytes") is not None]
        return {
            "schema": BLOCK_SCHEMA,
            "status": "passed" if not any(row["contaminated"] for row in rounds) else "contaminated",
            "architecture": architecture,
            "method": method,
            "core_budget": core_budget,
            "dimension": dimension,
            "parameter_count": parameter_count,
            "timesteps": timesteps,
            "row_ids": list(CANONICAL_ROWS),
            "started_utc": started_utc,
            "finished_utc": _utc_now(),
            "wall_seconds": time.monotonic() - started,
            "workers": ready,
            "worker_identity_checks": identity_checks,
            "placement_checks": placement,
            "resource_before": resource_before,
            "resource_admission": admission,
            "resource_after": resource_snapshot(cpu_list(core_budget)),
            "rounds": rounds,
            "block_median_makespan_seconds": (
                statistics.median(row["makespan_seconds"] for row in rounds)
                if rounds
                else None
            ),
            "outputs": {str(key): {"value": value, "score": score} for key, (value, score) in outputs.items()},
            "max_aggregate_rss_bytes": max(
                [row["aggregate_rss_bytes"] for row in rounds]
                + ([sum(ready_rss)] if ready_rss else [0])
            ),
        }
    except BaseException:
        terminate_processes(processes)
        raise
    finally:
        for handle in logs:
            handle.close()


def _outputs_from_arm(arm: Mapping[str, Any]) -> dict[int, tuple[list[float], list[float]]]:
    return {
        int(key): (list(value["value"]), list(value["score"]))
        for key, value in arm["outputs"].items()
    }


def run_paired_block(
    *,
    core_budget: int,
    dimension: int,
    parameter_count: int,
    timesteps: int,
    block_dir: Path,
    order: Sequence[str],
    autodiff_reference_path: str,
) -> dict[str, Any]:
    arms = {}
    for architecture in order:
        arms[architecture] = run_arm(
            architecture=architecture,
            core_budget=core_budget,
            dimension=dimension,
            parameter_count=parameter_count,
            timesteps=timesteps,
            block_dir=block_dir / architecture,
        )
    parity = parity_summary(_outputs_from_arm(arms["sharded"]), _outputs_from_arm(arms["batch_native"]))
    if not parity["passed"]:
        status = "failed"
    elif any(arm["status"] == "contaminated" for arm in arms.values()):
        status = "contaminated"
    else:
        status = "passed"
    payload = {
        "schema": BLOCK_SCHEMA,
        "status": status,
        "core_budget": core_budget,
        "dimension": dimension,
        "parameter_count": parameter_count,
        "timesteps": timesteps,
        "order": list(order),
        "arms": arms,
        "parity": parity,
        "autodiff_reference_path": autodiff_reference_path,
    }
    _write_json(block_dir / "block.json", payload)
    return payload


def run_paired_block_with_retry(
    *,
    core_budget: int,
    dimension: int,
    parameter_count: int,
    timesteps: int,
    block_dir: Path,
    order: Sequence[str],
    autodiff_reference_path: str,
) -> dict[str, Any]:
    attempts = []
    accepted = None
    for attempt in (1, 2):
        attempt_dir = block_dir / f"attempt-{attempt}"
        payload = run_paired_block(
            core_budget=core_budget,
            dimension=dimension,
            parameter_count=parameter_count,
            timesteps=timesteps,
            block_dir=attempt_dir,
            order=order,
            autodiff_reference_path=autodiff_reference_path,
        )
        attempts.append(
            {
                "attempt": attempt,
                "status": payload["status"],
                "path": str((attempt_dir / "block.json").relative_to(REPO_ROOT)),
            }
        )
        if payload["status"] == "passed":
            accepted = payload
            break
        if payload["status"] != "contaminated":
            break
    result = dict(accepted or payload)
    result["attempts"] = attempts
    if accepted is None and payload["status"] == "contaminated":
        result["status"] = "failed_contamination_after_retry"
    _write_json(block_dir / "block.json", result)
    return result


def ensure_autodiff_reference(
    *,
    phase_root: Path,
    dimension: int,
    parameter_count: int,
    timesteps: int,
) -> str:
    reference_dir = phase_root / f"d{dimension}-p{parameter_count}-t{timesteps}-autodiff-reference"
    reference_path = reference_dir / "reference.json"
    existing = _read_json(reference_path)
    if existing is not None and existing.get("status") == "passed":
        return str(reference_path.relative_to(REPO_ROOT))
    analytical = run_arm(
        architecture="batch_native",
        core_budget=1,
        dimension=dimension,
        parameter_count=parameter_count,
        timesteps=timesteps,
        block_dir=reference_dir / "analytical",
        method="analytical",
        measured_rounds=0,
    )
    autodiff = run_arm(
        architecture="batch_native",
        core_budget=1,
        dimension=dimension,
        parameter_count=parameter_count,
        timesteps=timesteps,
        block_dir=reference_dir / "autodiff",
        method="autodiff",
        measured_rounds=0,
    )
    parity = parity_summary(_outputs_from_arm(analytical), _outputs_from_arm(autodiff))
    payload = {
        "schema": BLOCK_SCHEMA,
        "status": "passed" if parity["passed"] else "failed",
        "dimension": dimension,
        "parameter_count": parameter_count,
        "timesteps": timesteps,
        "analytical": analytical,
        "autodiff": autodiff,
        "parity": parity,
    }
    _write_json(reference_path, payload)
    if payload["status"] != "passed":
        raise RuntimeError(f"analytical/autodiff parity failed: {reference_path}")
    return str(reference_path.relative_to(REPO_ROOT))


def _phase_specs(phase: str) -> tuple[list[tuple[int, int, int]], tuple[int, ...], int]:
    if phase == "smoke":
        return [(2, 3, 4)], (1, 2), 1
    if phase == "canary":
        return [(10, 50, 120)], (1, 4, 16), 1
    if phase == "nominate":
        return [(20, 150, 120)], CORE_BUDGETS, 5
    raise ValueError(f"phase does not have fixed specs: {phase}")


def _balanced_order(index: int) -> tuple[str, str]:
    return ("batch_native", "sharded") if index % 2 == 0 else ("sharded", "batch_native")


def _phase_result(blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    passed = [block for block in blocks if block.get("status") == "passed"]
    groups: dict[tuple[int, int, int, int], list[Mapping[str, Any]]] = {}
    for block in passed:
        key = (block["dimension"], block["parameter_count"], block["timesteps"], block["core_budget"])
        groups.setdefault(key, []).append(block)
    summaries = []
    for key, rows in sorted(groups.items()):
        batch = [row["arms"]["batch_native"]["block_median_makespan_seconds"] for row in rows]
        sharded = [row["arms"]["sharded"]["block_median_makespan_seconds"] for row in rows]
        summary = {
            "dimension": key[0],
            "parameter_count": key[1],
            "timesteps": key[2],
            "core_budget": key[3],
            "block_count": len(rows),
            "batch_native_median_seconds": statistics.median(batch),
            "sharded_median_seconds": statistics.median(sharded),
            "descriptive_sharded_over_batch_ratio": statistics.median(sharded) / statistics.median(batch),
        }
        if len(rows) >= 2:
            summary["paired_statistics_sharded_over_batch"] = paired_statistics(sharded, batch)
        summaries.append(summary)
    return {"passed_block_count": len(passed), "total_block_count": len(blocks), "summaries": summaries}


def _load_or_new(output_root: Path) -> dict[str, Any]:
    path = output_root / "status.json"
    existing = _read_json(path)
    manifest = source_manifest()
    if existing is None:
        return {
            "schema": SCHEMA,
            "status": "initialized",
            "created_utc": _utc_now(),
            "source_manifest": manifest,
            "topology": topology_contract(),
            "phases": {},
            "plan": PLAN,
            "result": RESULT,
        }
    if existing.get("schema") != SCHEMA or existing.get("source_manifest", {}).get("fingerprint") != manifest["fingerprint"]:
        raise RuntimeError("existing status source or schema drift")
    return existing


def run_phase(phase: str, output_root: Path) -> dict[str, Any]:
    status = _load_or_new(output_root)
    if status["topology"].get("valid") is not True:
        raise RuntimeError("CPU topology contract failed")
    if phase in status["phases"] and status["phases"][phase].get("status") == "complete":
        return status["phases"][phase]
    workloads, budgets, repetitions = _phase_specs(phase)
    phase_root = output_root / phase
    status["status"] = "running"
    status["active_phase"] = phase
    status["updated_utc"] = _utc_now()
    _write_json(output_root / "status.json", status)
    blocks = []
    try:
        for dimension, parameter_count, timesteps in workloads:
            autodiff_reference_path = ensure_autodiff_reference(
                phase_root=phase_root,
                dimension=dimension,
                parameter_count=parameter_count,
                timesteps=timesteps,
            )
            for core_budget in budgets:
                for block_index in range(repetitions):
                    block_dir = phase_root / f"d{dimension}-p{parameter_count}-t{timesteps}-k{core_budget}-block{block_index:02d}"
                    existing = _read_json(block_dir / "block.json")
                    if existing is not None and existing.get("status") == "passed":
                        blocks.append(existing)
                        continue
                    block = run_paired_block_with_retry(
                        core_budget=core_budget,
                        dimension=dimension,
                        parameter_count=parameter_count,
                        timesteps=timesteps,
                        block_dir=block_dir,
                        order=_balanced_order(block_index + core_budget),
                        autodiff_reference_path=autodiff_reference_path,
                    )
                    blocks.append(block)
                    if block["status"] != "passed":
                        raise RuntimeError(f"phase block failed: {block_dir}")
    except BaseException as exc:
        status = _load_or_new(output_root)
        status["status"] = "failed"
        status["active_phase"] = phase
        status["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        status["updated_utc"] = _utc_now()
        _write_json(output_root / "status.json", status)
        raise
    result = _phase_result(blocks)
    phase_status = {
        "status": "complete",
        "finished_utc": _utc_now(),
        "result": result,
        "block_paths": [str(path.relative_to(REPO_ROOT)) for path in sorted(phase_root.glob("*/block.json"))],
    }
    status["phases"][phase] = phase_status
    status["status"] = "running"
    status.pop("active_phase", None)
    status.pop("failure", None)
    status["updated_utc"] = _utc_now()
    _write_json(output_root / "status.json", status)
    return phase_status


def nominate(output_root: Path) -> dict[str, Any]:
    phase = run_phase("nominate", output_root)
    summaries = phase["result"]["summaries"]
    candidates = []
    for row in summaries:
        for architecture in ("batch_native", "sharded"):
            candidates.append(
                {
                    "architecture": architecture,
                    "core_budget": row["core_budget"],
                    "median_seconds": row[f"{architecture}_median_seconds"],
                }
            )
    candidates.sort(key=lambda row: (row["median_seconds"], row["core_budget"], row["architecture"]))
    best = candidates[0]
    near = [row for row in candidates if row["median_seconds"] <= 1.05 * best["median_seconds"]]
    nominee = min(near, key=lambda row: (row["core_budget"], row["median_seconds"], row["architecture"]))
    status = _load_or_new(output_root)
    status["nomination"] = {"status": "nominated_descriptive_only", "candidate": nominee, "all_candidates": candidates}
    _write_json(output_root / "status.json", status)
    return status["nomination"]


def confirm(output_root: Path) -> dict[str, Any]:
    status = _load_or_new(output_root)
    nomination = status.get("nomination") or nominate(output_root)
    core_budget = int(nomination["candidate"]["core_budget"])
    workloads = ((30, 150, 120), (10, 50, 120), (30, 50, 120))
    blocks = []
    phase_root = output_root / "confirm"
    rng = random.Random(RANDOM_SEED)
    orders = []
    for workload_index, (dimension, parameter_count, timesteps) in enumerate(workloads):
        autodiff_reference_path = ensure_autodiff_reference(
            phase_root=phase_root,
            dimension=dimension,
            parameter_count=parameter_count,
            timesteps=timesteps,
        )
        base_orders = [
            ("batch_native", "sharded") if index < 6 else ("sharded", "batch_native")
            for index in range(12)
        ]
        rng.shuffle(base_orders)
        orders.append(base_orders)
        for block_index, order in enumerate(base_orders):
            block_dir = phase_root / f"d{dimension}-p{parameter_count}-t{timesteps}-k{core_budget}-block{block_index:02d}"
            existing = _read_json(block_dir / "block.json")
            if existing is not None and existing.get("status") == "passed":
                blocks.append(existing)
                continue
            block = run_paired_block_with_retry(
                core_budget=core_budget,
                dimension=dimension,
                parameter_count=parameter_count,
                timesteps=timesteps,
                block_dir=block_dir,
                order=order,
                autodiff_reference_path=autodiff_reference_path,
            )
            blocks.append(block)
            if block["status"] != "passed":
                raise RuntimeError(f"confirmation block failed: {block_dir}")
    result = _phase_result(blocks)
    nominee_architecture = nomination["candidate"]["architecture"]
    decisions = []
    for row in result["summaries"]:
        stats = row["paired_statistics_sharded_over_batch"]
        if nominee_architecture == "batch_native":
            interval = [1.0 / stats["bootstrap_95_interval"][1], 1.0 / stats["bootstrap_95_interval"][0]]
        else:
            interval = stats["bootstrap_95_interval"]
        decisions.append({"workload": [row["dimension"], row["parameter_count"], row["timesteps"]], "nominee_over_comparator_interval": interval})
    primary_row = next(
        row for row in decisions if row["workload"] == [30, 150, 120]
    )
    primary = primary_row["nominee_over_comparator_interval"]
    recommend = primary[1] < 0.95 and all(row["nominee_over_comparator_interval"][1] <= 1.05 for row in decisions)
    final_decision = (
        "RECOMMEND_PERSISTENT_SHARDING_FOR_TESTED_CPU_THROUGHPUT"
        if recommend and nominee_architecture == "sharded"
        else "RECOMMEND_BATCH_NATIVE_FOR_TESTED_CPU_THROUGHPUT"
        if recommend
        else "CPU_ARCHITECTURES_STATISTICALLY_UNRESOLVED"
    )
    phase_status = {"status": "complete", "finished_utc": _utc_now(), "result": result, "decision_checks": decisions, "decision": final_decision, "orders": orders}
    status = _load_or_new(output_root)
    status["phases"]["confirm"] = phase_status
    status["status"] = "complete"
    status["decision"] = final_decision
    status["finished_utc"] = _utc_now()
    _write_json(output_root / "status.json", status)
    return phase_status


def _self_check() -> int:
    checks = {
        "canonical_rows": CANONICAL_ROWS == tuple(range(16)),
        "partitions": all(sorted(row for part in partition_rows(k) for row in part) == list(CANONICAL_ROWS) for k in CORE_BUDGETS),
        "cpu_lists": all(cpu_list(k) == PHYSICAL_CPU_POOL[:k] for k in CORE_BUDGETS),
        "source_manifest": len(source_manifest()["fingerprint"]) == 64,
        "topology": topology_contract()["valid"],
    }
    print(json.dumps(checks, indent=2, sort_keys=True))
    return 0 if all(checks.values()) else 1


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "canary", "nominate", "confirm"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-mode", choices=("batch", "sharded"))
    parser.add_argument("--method", choices=("analytical", "autodiff"), default="analytical")
    parser.add_argument("--row-ids", default="")
    parser.add_argument("--intra", type=int, default=1)
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--parameter-count", type=int, default=3)
    parser.add_argument("--timesteps", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.worker:
        return _worker_main(args)
    if args.self_check:
        return _self_check()
    if args.phase is None:
        raise ValueError("--phase is required")
    output_root = args.output_root.resolve()
    output_root.relative_to(REPO_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)
    if args.phase == "confirm":
        print(json.dumps(confirm(output_root), indent=2, sort_keys=True))
    elif args.phase == "nominate":
        print(json.dumps(nominate(output_root), indent=2, sort_keys=True))
    else:
        print(json.dumps(run_phase(args.phase, output_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
