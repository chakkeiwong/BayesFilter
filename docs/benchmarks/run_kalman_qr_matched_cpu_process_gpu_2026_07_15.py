#!/usr/bin/env python
"""Matched analytical Kalman QR benchmark for CPU batch, processes, and GPU."""

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
PLAN = "docs/plans/bayesfilter-kalman-qr-matched-cpu-process-gpu-comparison-plan-2026-07-15.md"
RESULT = "docs/plans/bayesfilter-kalman-qr-matched-cpu-process-gpu-comparison-result-2026-07-15.md"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "docs/benchmarks/kalman_qr_matched_cpu_process_gpu_2026-07-15"
SCHEMA = "bayesfilter.kalman_qr.matched_cpu_process_gpu.v1"
WORKER_SCHEMA = "bayesfilter.kalman_qr.matched_worker.v1"
ARMS = ("cpu_native_b16_xla", "cpu_processes_16xb1_xla", "gpu_native_b16_xla")
CPU_POOL = tuple(range(16, 32))
ROW_IDS = tuple(range(16))
DIMENSION = 30
PARAMETER_COUNT = 150
TIMESTEPS = 12
BLOCKS = 6
WARM_CALLS = 2
MEASURED_CALLS = 5
RANDOM_SEED = 20260715
MAX_CPU_RSS_BYTES = 16 * 1024**3
MAX_GPU_ALLOCATOR_BYTES = 16 * 1024**3
MAX_GPU_PRELAUNCH_UTILIZATION_PERCENT = 50
MAX_GPU_PRELAUNCH_MEMORY_MIB = 2048
MAX_CPU_PRELAUNCH_BUSY_FRACTION = 0.10
CPU_RESOURCE_WAIT_SECONDS = 600
READY_TIMEOUT_SECONDS = 300
ROUND_TIMEOUT_SECONDS = 60
PROCESS_TIMEOUT_SECONDS = 900
GPU_XLA_FLAG = "--xla_gpu_enable_triton_gemm=false"
GPU_TRUST_BASIS = "owner_designated_managed_session_visible_gpu_trusted"
FLOAT32_PARITY = {"value_rtol": 2e-4, "value_atol": 2e-4, "score_rtol": 2e-4, "score_atol": 2e-4}
SOURCE_PATHS = (
    "bayesfilter/linear/kalman_qr_tf.py",
    "bayesfilter/linear/kalman_qr_derivatives_tf.py",
    "bayesfilter/linear/qr_factor_tf.py",
    "scripts/benchmark_kalman_qr_parameter_count_scaling.py",
    "docs/benchmarks/run_kalman_qr_matched_cpu_process_gpu_2026_07_15.py",
    PLAN,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest() -> dict[str, Any]:
    files = [{"path": path, "sha256": _sha256(REPO_ROOT / path)} for path in SOURCE_PATHS]
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"files": files, "fingerprint": hashlib.sha256(encoded).hexdigest()}


def _parse_csv_rows(text: str) -> list[list[str]]:
    return [[field.strip() for field in line.split(",")] for line in text.splitlines() if line.strip()]


def gpu_snapshot() -> dict[str, Any]:
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"],
        check=False, capture_output=True, text=True, timeout=15,
    )
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        check=False, capture_output=True, text=True, timeout=15,
    )
    return {"observed_utc": _utc_now(), "gpu_returncode": gpu.returncode, "app_returncode": apps.returncode, "gpus": _parse_csv_rows(gpu.stdout), "compute_apps": _parse_csv_rows(apps.stdout)}


def gpu0_admissible(snapshot: Mapping[str, Any]) -> bool:
    rows = snapshot.get("gpus", [])
    if snapshot.get("gpu_returncode") != 0 or not rows or len(rows[0]) != 7 or rows[0][0] != "0":
        return False
    try:
        used = int(rows[0][4])
        utilization = int(rows[0][5])
    except (TypeError, ValueError):
        return False
    return used <= MAX_GPU_PRELAUNCH_MEMORY_MIB and utilization < MAX_GPU_PRELAUNCH_UTILIZATION_PERCENT


def gpu0_compute_pids(snapshot: Mapping[str, Any]) -> set[int] | None:
    rows = snapshot.get("gpus", [])
    apps = snapshot.get("compute_apps", [])
    if snapshot.get("gpu_returncode") != 0 or snapshot.get("app_returncode") != 0 or not rows or len(rows[0]) < 2:
        return None
    uuid = rows[0][1]
    try:
        return {int(row[1]) for row in apps if len(row) >= 2 and row[0] == uuid}
    except (TypeError, ValueError):
        return None


def topology_contract() -> dict[str, Any]:
    completed = subprocess.run(["lscpu", "-e=CPU,NODE,SOCKET,CORE,ONLINE"], check=False, capture_output=True, text=True, timeout=15)
    parsed = []
    for line in completed.stdout.splitlines()[1:]:
        fields = line.split()
        if len(fields) == 5:
            parsed.append({"cpu": int(fields[0]), "node": int(fields[1]), "socket": int(fields[2]), "core": int(fields[3]), "online": fields[4].lower() == "yes"})
    by_cpu = {row["cpu"]: row for row in parsed}
    selected = [by_cpu.get(cpu) for cpu in CPU_POOL]
    valid = completed.returncode == 0 and all(row is not None and row["online"] and row["node"] == 0 and row["socket"] == 0 for row in selected) and len({row["core"] for row in selected if row}) == 16
    return {"returncode": completed.returncode, "selected": selected, "valid": valid}


def _read_rss_bytes(pid: int) -> int | None:
    try:
        pages = int(Path(f"/proc/{pid}/statm").read_text(encoding="utf-8").split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


def _process_cpu_seconds(pid: int) -> float | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        return (int(fields[13]) + int(fields[14])) / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return None


def _cpu_stat_snapshot(cpus: Sequence[int]) -> dict[int, tuple[int, int]]:
    selected = set(cpus)
    result = {}
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


def cpu_busy_fractions(before: Mapping[int, tuple[int, int]], after: Mapping[int, tuple[int, int]]) -> dict[int, float] | None:
    if set(before) != set(after) or not before:
        return None
    result = {}
    for cpu in before:
        total = after[cpu][0] - before[cpu][0]
        idle = after[cpu][1] - before[cpu][1]
        if total <= 0:
            return None
        result[cpu] = max(0.0, min(1.0, (total - idle) / total))
    return result


def target_cpu_contamination_seconds(before: Mapping[int, tuple[int, int]], after: Mapping[int, tuple[int, int]], owned_cpu_seconds: float) -> float | None:
    if set(before) != set(after) or not before:
        return None
    busy_ticks = sum((after[cpu][0] - before[cpu][0]) - (after[cpu][1] - before[cpu][1]) for cpu in before)
    return max(0.0, busy_ticks / os.sysconf("SC_CLK_TCK") - owned_cpu_seconds)


def wait_for_cpu_pool(*, wait_seconds: float = CPU_RESOURCE_WAIT_SECONDS, sample_seconds: float = 1.0) -> dict[str, Any]:
    started = time.monotonic()
    while True:
        before = _cpu_stat_snapshot(CPU_POOL)
        time.sleep(sample_seconds)
        after = _cpu_stat_snapshot(CPU_POOL)
        busy = cpu_busy_fractions(before, after)
        sample = {"observed_utc": _utc_now(), "sample_seconds": sample_seconds, "busy_fractions": None if busy is None else {str(cpu): value for cpu, value in busy.items()}, "load_average": list(os.getloadavg())}
        if busy is not None and max(busy.values(), default=1.0) < MAX_CPU_PRELAUNCH_BUSY_FRACTION:
            return sample
        if time.monotonic() - started >= wait_seconds:
            raise TimeoutError(f"CPU pool did not become idle: {sample}")
        time.sleep(5)


def _numa_fraction_node0() -> float | None:
    totals: dict[int, int] = {}
    try:
        lines = Path("/proc/self/numa_maps").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        if "anon=" not in line:
            continue
        for field in line.split():
            if field.startswith("N") and "=" in field:
                node, pages = field.split("=", 1)
                if node[1:].isdigit() and pages.isdigit():
                    totals[int(node[1:])] = totals.get(int(node[1:]), 0) + int(pages)
    total = sum(totals.values())
    return totals.get(0, 0) / total if total else None


def _task_affinities() -> list[list[int]]:
    result = []
    for path in Path(f"/proc/{os.getpid()}/task").glob("*/status"):
        try:
            row = next(line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("Cpus_allowed_list:"))
        except (OSError, StopIteration):
            continue
        cpus: set[int] = set()
        for part in row.split(":", 1)[1].strip().split(","):
            if "-" in part:
                first, last = (int(value) for value in part.split("-", 1))
                cpus.update(range(first, last + 1))
            elif part:
                cpus.add(int(part))
        result.append(sorted(cpus))
    return result


def arm_worker_specs(arm: str) -> list[dict[str, Any]]:
    if arm == "cpu_native_b16_xla":
        return [{"row_ids": list(ROW_IDS), "cpus": list(CPU_POOL), "intra": 16, "device": "cpu"}]
    if arm == "cpu_processes_16xb1_xla":
        return [{"row_ids": [row], "cpus": [cpu], "intra": 1, "device": "cpu"} for row, cpu in zip(ROW_IDS, CPU_POOL, strict=True)]
    if arm == "gpu_native_b16_xla":
        return [{"row_ids": list(ROW_IDS), "cpus": [], "intra": 1, "device": "gpu"}]
    raise ValueError(f"unknown arm: {arm}")


def balanced_orders(block_count: int = BLOCKS, seed: int = RANDOM_SEED) -> list[list[str]]:
    orders = []
    for shift in range(block_count):
        orders.append([ARMS[(index + shift) % len(ARMS)] for index in range(len(ARMS))])
    random.Random(seed).shuffle(orders)
    return orders


def parity_summary(candidate: Mapping[int, tuple[Sequence[float], Sequence[float]]], reference: Mapping[int, tuple[Sequence[float], Sequence[float]]]) -> dict[str, Any]:
    if set(candidate) != set(ROW_IDS) or set(reference) != set(ROW_IDS):
        return {"passed": False, "reason": "row_identity_mismatch"}
    finite = True
    shapes = True
    max_value = max_score = allowed_value = allowed_score = 0.0
    for row in ROW_IDS:
        cv, cs = candidate[row]
        rv, rs = reference[row]
        shapes = shapes and len(cv) == len(rv) == 1 and len(cs) == len(rs) == PARAMETER_COUNT
        for left, right in zip(cv, rv, strict=False):
            finite = finite and math.isfinite(float(left)) and math.isfinite(float(right))
            max_value = max(max_value, abs(float(left) - float(right)))
            allowed_value = max(allowed_value, FLOAT32_PARITY["value_atol"] + FLOAT32_PARITY["value_rtol"] * abs(float(right)))
        for left, right in zip(cs, rs, strict=False):
            finite = finite and math.isfinite(float(left)) and math.isfinite(float(right))
            max_score = max(max_score, abs(float(left) - float(right)))
            allowed_score = max(allowed_score, FLOAT32_PARITY["score_atol"] + FLOAT32_PARITY["score_rtol"] * abs(float(right)))
    return {"passed": finite and shapes and max_value <= allowed_value and max_score <= allowed_score, "finite": finite, "shapes": shapes, "max_value_residual": max_value, "max_score_residual": max_score, "max_allowed_value_residual": allowed_value, "max_allowed_score_residual": allowed_score, "tolerances": FLOAT32_PARITY}


def paired_statistics(candidate: Sequence[float], comparator: Sequence[float], *, seed: int = RANDOM_SEED, resamples: int = 10000) -> dict[str, Any]:
    if len(candidate) != len(comparator) or not candidate or any(value <= 0 for value in (*candidate, *comparator)):
        raise ValueError("paired positive equal-length timings required")
    logs = [math.log(left / right) for left, right in zip(candidate, comparator, strict=True)]
    rng = random.Random(seed)
    boot = sorted(statistics.fmean(logs[rng.randrange(len(logs))] for _ in logs) for _ in range(resamples))
    positives = sum(value > 0 for value in logs if value != 0)
    n = sum(value != 0 for value in logs)
    tail = sum(math.comb(n, index) for index in range(min(positives, n - positives) + 1)) / 2**n if n else 0.5
    return {"paired_block_count": len(logs), "geometric_mean_ratio": math.exp(statistics.fmean(logs)), "bootstrap_95_interval": [math.exp(boot[int(0.025 * resamples)]), math.exp(boot[min(resamples - 1, int(0.975 * resamples))])], "sign_test_two_sided_p": min(1.0, 2 * tail), "bootstrap_resamples": resamples, "seed": seed}


def _wait_until(start_ns: int) -> None:
    while True:
        remaining = start_ns - time.monotonic_ns()
        if remaining <= 0:
            return
        if remaining > 2_000_000:
            time.sleep((remaining - 1_000_000) / 1e9)


def _worker_main(args: argparse.Namespace) -> int:
    import tensorflow as tf

    sys.path.insert(0, str(REPO_ROOT))
    from scripts.benchmark_kalman_qr_parameter_count_scaling import _make_parameter_cloud, _synchronize_outputs, build_batch_native_analytic_fn, make_fixture

    started = time.perf_counter()
    tf.config.threading.set_intra_op_parallelism_threads(args.intra)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    if args.device == "gpu":
        physical = tf.config.list_physical_devices("GPU")
        if len(physical) != 1:
            raise RuntimeError(f"expected exactly one visible physical GPU, got {physical}")
        tf.config.experimental.set_memory_growth(physical[0], True)
        if tf.config.experimental.get_memory_growth(physical[0]) is not True:
            raise RuntimeError("GPU memory growth is not enabled")
        logical = tf.config.list_logical_devices("GPU")
        if len(logical) != 1:
            raise RuntimeError(f"expected exactly one logical GPU, got {logical}")
        selected_device = "/GPU:0"
    else:
        if tf.config.list_physical_devices("GPU"):
            raise RuntimeError("CPU worker can see a GPU")
        physical, logical, selected_device = [], [], "/CPU:0"
    fixture = make_fixture(DIMENSION, PARAMETER_COUNT, TIMESTEPS, dtype=tf.float32)
    cloud = _make_parameter_cloud(fixture)
    row_ids = [int(value) for value in args.row_ids.split(",")]
    parameters = tf.gather(cloud, row_ids, axis=0)
    with tf.device(selected_device):
        concrete = build_batch_native_analytic_fn(fixture, batch_size=len(row_ids), jit_compile=True).get_concrete_function(parameters)

    def invoke():
        with tf.device(selected_device):
            outputs = concrete(parameters)
        _synchronize_outputs(outputs)
        return outputs

    first_started = time.perf_counter()
    outputs = invoke()
    first_seconds = time.perf_counter() - first_started
    warm = []
    for _ in range(WARM_CALLS):
        tick = time.perf_counter()
        outputs = invoke()
        warm.append(time.perf_counter() - tick)
    numa_fraction = _numa_fraction_node0() if args.device == "cpu" else None
    allocator = tf.config.experimental.get_memory_info("GPU:0") if args.device == "gpu" else None
    ready = {"schema": WORKER_SCHEMA, "event": "ready", "pid": os.getpid(), "row_ids": row_ids, "device": args.device, "selected_device": selected_device, "jit_compile": True, "xla_flags": os.environ.get("XLA_FLAGS", "UNSET"), "affinity": sorted(os.sched_getaffinity(0)), "task_affinities": _task_affinities(), "numa_node0_fraction": numa_fraction, "rss_bytes": _read_rss_bytes(os.getpid()), "cold_time_to_ready_seconds": time.perf_counter() - started, "first_executable_call_seconds": first_seconds, "warm_seconds": warm, "physical_gpus": [value.name for value in physical], "logical_gpus": [value.name for value in logical], "gpu_memory_growth": bool(args.device == "gpu"), "gpu_allocator_memory": None if allocator is None else {"current_bytes": int(allocator["current"]), "peak_bytes": int(allocator["peak"])}}
    print(json.dumps(ready, allow_nan=False), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request["command"] == "stop":
            print(json.dumps({"event": "stopped", "pid": os.getpid()}), flush=True)
            return 0
        if request["command"] == "output":
            outputs = invoke()
            value, score = outputs
            print(json.dumps({"event": "output", "pid": os.getpid(), "row_ids": row_ids, "value": value.numpy().tolist(), "score": score.numpy().tolist()}, allow_nan=False), flush=True)
            continue
        if request["command"] == "run":
            _wait_until(int(request["start_ns"]))
            tick = time.perf_counter()
            outputs = invoke()
            elapsed = time.perf_counter() - tick
            allocator = tf.config.experimental.get_memory_info("GPU:0") if args.device == "gpu" else None
            print(json.dumps({"event": "done", "pid": os.getpid(), "round": request["round"], "kernel_seconds": elapsed, "rss_bytes": _read_rss_bytes(os.getpid()), "gpu_allocator_memory": None if allocator is None else {"current_bytes": int(allocator["current"]), "peak_bytes": int(allocator["peak"])}}, allow_nan=False), flush=True)
            continue
        raise RuntimeError(f"unknown command: {request}")
    return 0


def _worker_command(spec: Mapping[str, Any]) -> list[str]:
    command = [str(PYTHON), str(SCRIPT_PATH), "--worker", "--device", str(spec["device"]), "--intra", str(spec["intra"]), "--row-ids", ",".join(str(value) for value in spec["row_ids"])]
    if spec["device"] == "cpu":
        return [str(HWLOC_BIND), "--membind", "node:0", "--", "taskset", "-c", ",".join(str(value) for value in spec["cpus"]), *command]
    return command


def worker_environment(spec: Mapping[str, Any]) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update({"CUDA_VISIBLE_DEVICES": "-1" if spec["device"] == "cpu" else "0", "TF_FORCE_GPU_ALLOW_GROWTH": "true", "OMP_NUM_THREADS": str(spec["intra"]), "TF_NUM_INTRAOP_THREADS": str(spec["intra"]), "TF_NUM_INTEROP_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    if spec["device"] == "gpu":
        environment["XLA_FLAGS"] = GPU_XLA_FLAG
    else:
        environment.pop("XLA_FLAGS", None)
    return environment


def _read_messages(processes: Sequence[subprocess.Popen[str]], timeout: float) -> list[dict[str, Any]]:
    selector = selectors.DefaultSelector()
    by_fd = {}
    for process in processes:
        assert process.stdout is not None
        selector.register(process.stdout, selectors.EVENT_READ)
        by_fd[process.stdout.fileno()] = process
    messages = []
    pending = set(processes)
    deadline = time.monotonic() + timeout
    try:
        while pending:
            events = selector.select(max(0.0, deadline - time.monotonic()))
            if not events:
                raise TimeoutError("worker response timeout")
            for key, _mask in events:
                process = by_fd[key.fileobj.fileno()]
                line = key.fileobj.readline()
                if not line:
                    raise RuntimeError(f"worker exited before response: {process.pid}")
                messages.append(json.loads(line))
                pending.discard(process)
    finally:
        selector.close()
    return messages


def _send(processes: Sequence[subprocess.Popen[str]], payload: Mapping[str, Any]) -> None:
    text = json.dumps(payload) + "\n"
    for process in processes:
        if process.stdin is None:
            raise RuntimeError("worker stdin unavailable")
        process.stdin.write(text)
        process.stdin.flush()


def terminate_processes(processes: Sequence[subprocess.Popen[str]]) -> None:
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 10
    for process in processes:
        try:
            process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)


def _collect_outputs(messages: Sequence[Mapping[str, Any]]) -> dict[int, tuple[list[float], list[float]]]:
    result = {}
    for message in messages:
        for row, value, score in zip(message["row_ids"], message["value"], message["score"], strict=True):
            if int(row) in result:
                raise RuntimeError("duplicate output row")
            result[int(row)] = ([float(value)], [float(item) for item in score])
    if set(result) != set(ROW_IDS):
        raise RuntimeError("output rows do not cover 0..15")
    return result


def run_arm(arm: str, block_index: int, output_root: Path) -> dict[str, Any]:
    specs = arm_worker_specs(arm)
    processes: list[subprocess.Popen[str]] = []
    logs = []
    gpu_before = gpu_snapshot() if arm == "gpu_native_b16_xla" else None
    if gpu_before is not None and not gpu0_admissible(gpu_before):
        raise RuntimeError(f"GPU 0 prelaunch admission failed: {gpu_before}")
    baseline_gpu_pids = gpu0_compute_pids(gpu_before) if gpu_before is not None else None
    if gpu_before is not None and baseline_gpu_pids is None:
        raise RuntimeError("GPU 0 prelaunch process census failed")
    cpu_admission = wait_for_cpu_pool() if arm != "gpu_native_b16_xla" else None
    started = time.perf_counter()
    try:
        for index, spec in enumerate(specs):
            log_path = output_root / f"block-{block_index:02d}" / arm / f"worker-{index}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(_worker_command(spec), cwd=REPO_ROOT, env=worker_environment(spec), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=handle, text=True, start_new_session=True, bufsize=1)
            processes.append(process)
            logs.append(handle)
        ready = _read_messages(processes, READY_TIMEOUT_SECONDS)
        ready_by_pid = {row["pid"]: row for row in ready}
        valid_ready = []
        for process, spec in zip(processes, specs, strict=True):
            row = ready_by_pid.get(process.pid, {})
            valid = row.get("schema") == WORKER_SCHEMA and row.get("row_ids") == spec["row_ids"] and row.get("device") == spec["device"] and row.get("jit_compile") is True
            if spec["device"] == "cpu":
                valid = valid and row.get("selected_device") == "/CPU:0" and row.get("affinity") == spec["cpus"] and bool(row.get("task_affinities")) and all(cpus == spec["cpus"] for cpus in row["task_affinities"]) and float(row.get("numa_node0_fraction", 0)) >= 0.95 and row.get("physical_gpus") == []
            else:
                valid = valid and row.get("selected_device") == "/GPU:0" and row.get("gpu_memory_growth") is True and row.get("xla_flags") == GPU_XLA_FLAG and len(row.get("physical_gpus", [])) == 1 and len(row.get("logical_gpus", [])) == 1
            valid_ready.append(valid)
        if not all(valid_ready):
            raise RuntimeError(f"worker readiness contract failed: {ready}")
        gpu_runtime_samples = []
        allowed_gpu_pids = set(baseline_gpu_pids or set()).union(process.pid for process in processes)
        if arm == "gpu_native_b16_xla":
            sample = gpu_snapshot()
            gpu_runtime_samples.append(sample)
            if gpu0_compute_pids(sample) is None or not gpu0_compute_pids(sample).issubset(allowed_gpu_pids):
                raise RuntimeError(f"foreign GPU process present after readiness: {sample}")
        _send(processes, {"command": "output"})
        outputs = _collect_outputs(_read_messages(processes, ROUND_TIMEOUT_SECONDS))
        rounds = []
        for round_index in range(MEASURED_CALLS):
            target_cpu_before = _cpu_stat_snapshot(CPU_POOL) if arm != "gpu_native_b16_xla" else {}
            cpu_before = {process.pid: _process_cpu_seconds(process.pid) for process in processes}
            start_ns = time.monotonic_ns() + 50_000_000
            _send(processes, {"command": "run", "round": round_index, "start_ns": start_ns})
            messages = _read_messages(processes, ROUND_TIMEOUT_SECONDS)
            supervisor_wall = max(0.0, (time.monotonic_ns() - start_ns) / 1e9)
            cpu_after = {process.pid: _process_cpu_seconds(process.pid) for process in processes}
            target_cpu_after = _cpu_stat_snapshot(CPU_POOL) if arm != "gpu_native_b16_xla" else {}
            owned_cpu = sum(max(0.0, float(cpu_after[pid]) - float(cpu_before[pid])) for pid in cpu_before if cpu_before[pid] is not None and cpu_after[pid] is not None)
            kernel_makespan = max(float(row["kernel_seconds"]) for row in messages)
            contamination = target_cpu_contamination_seconds(target_cpu_before, target_cpu_after, owned_cpu) if arm != "gpu_native_b16_xla" else None
            contamination_threshold = max(0.25, 0.02 * len(CPU_POOL) * kernel_makespan)
            rss = sum(value for value in (_read_rss_bytes(process.pid) for process in processes) if value is not None)
            allocator_peaks = [row.get("gpu_allocator_memory", {}).get("peak_bytes") for row in messages if row.get("gpu_allocator_memory")]
            allocator_peak = max(allocator_peaks) if allocator_peaks else None
            if rss > MAX_CPU_RSS_BYTES:
                raise MemoryError("aggregate process RSS exceeded 16 GiB")
            if allocator_peak is not None and allocator_peak > MAX_GPU_ALLOCATOR_BYTES:
                raise MemoryError("GPU allocator peak exceeded 16 GiB")
            if arm != "gpu_native_b16_xla" and (contamination is None or contamination > contamination_threshold):
                raise RuntimeError(f"target CPU contamination exceeded threshold: observed={contamination}, threshold={contamination_threshold}")
            rounds.append({"round": round_index, "kernel_makespan_seconds": kernel_makespan, "worker_kernel_seconds": [float(row["kernel_seconds"]) for row in messages], "supervisor_dispatch_seconds": supervisor_wall, "owned_cpu_seconds": owned_cpu, "average_cpu_cores_during_kernel": owned_cpu / kernel_makespan if kernel_makespan else None, "unattributed_target_cpu_seconds": contamination, "contamination_threshold_seconds": contamination_threshold if arm != "gpu_native_b16_xla" else None, "aggregate_rss_bytes": rss, "gpu_allocator_peak_bytes": allocator_peak})
            if arm == "gpu_native_b16_xla":
                sample = gpu_snapshot()
                gpu_runtime_samples.append(sample)
                pids = gpu0_compute_pids(sample)
                if pids is None or not pids.issubset(allowed_gpu_pids):
                    raise RuntimeError(f"foreign GPU process overlapped measured calls: {sample}")
        _send(processes, {"command": "stop"})
        _read_messages(processes, 30)
        for process in processes:
            process.wait(timeout=30)
        if any(process.returncode != 0 for process in processes):
            raise RuntimeError("worker returned nonzero")
        gpu_after = gpu_snapshot() if arm == "gpu_native_b16_xla" else None
        if gpu_after is not None and gpu0_compute_pids(gpu_after) != baseline_gpu_pids:
            raise RuntimeError(f"GPU process census did not return to baseline: {gpu_after}")
        return {"arm": arm, "block_index": block_index, "status": "passed", "finished_utc": _utc_now(), "wall_seconds_including_cold": time.perf_counter() - started, "max_worker_cold_to_ready_seconds": max(float(row["cold_time_to_ready_seconds"]) for row in ready), "workers": ready, "readiness_checks": valid_ready, "outputs": {str(row): {"value": value, "score": score} for row, (value, score) in outputs.items()}, "rounds": rounds, "block_median_kernel_seconds": statistics.median(row["kernel_makespan_seconds"] for row in rounds), "max_aggregate_rss_bytes": max(row["aggregate_rss_bytes"] for row in rounds), "max_gpu_allocator_peak_bytes": max((row["gpu_allocator_peak_bytes"] or 0) for row in rounds), "cpu_admission": cpu_admission, "gpu_before": gpu_before, "gpu_runtime_samples": gpu_runtime_samples, "gpu_after": gpu_after}
    finally:
        terminate_processes(processes)
        for handle in logs:
            handle.close()


def _outputs(arm: Mapping[str, Any]) -> dict[int, tuple[list[float], list[float]]]:
    return {int(row): (value["value"], value["score"]) for row, value in arm["outputs"].items()}


def _git_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=False, capture_output=True, text=True, timeout=15)
    return result.stdout.strip() if result.returncode == 0 else "UNAVAILABLE"


def run(output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    topology = topology_contract()
    if not topology["valid"]:
        raise RuntimeError(f"CPU topology contract failed: {topology}")
    status = {"schema": SCHEMA, "status": "running", "created_utc": _utc_now(), "plan": PLAN, "result": RESULT, "source_manifest": source_manifest(), "topology": topology, "configuration": {"dimension": DIMENSION, "parameter_count": PARAMETER_COUNT, "timesteps": TIMESTEPS, "batch_size": 16, "dtype": "float32", "jit_compile": True, "warm_calls": WARM_CALLS, "measured_calls": MEASURED_CALLS, "fresh_blocks": BLOCKS, "orders": balanced_orders()}, "blocks": []}
    _write_json(output_root / "status.json", status)
    try:
        for block_index, order in enumerate(status["configuration"]["orders"]):
            block = {"block_index": block_index, "order": order, "arms": {}, "parity": {}}
            for arm in order:
                arm_result = run_arm(arm, block_index, output_root)
                block["arms"][arm] = arm_result
                _write_json(output_root / f"block-{block_index:02d}" / arm / "result.json", arm_result)
            reference = _outputs(block["arms"]["cpu_native_b16_xla"])
            for arm in ARMS:
                block["parity"][arm] = parity_summary(_outputs(block["arms"][arm]), reference)
            if not all(row["passed"] for row in block["parity"].values()):
                raise RuntimeError(f"cross-arm parity failed in block {block_index}")
            block["status"] = "passed"
            status["blocks"].append(block)
            _write_json(output_root / "status.json", status)
        arm_summaries = {}
        for arm in ARMS:
            medians = [block["arms"][arm]["block_median_kernel_seconds"] for block in status["blocks"]]
            arm_summaries[arm] = {"block_median_seconds": medians, "median_seconds": statistics.median(medians), "proposals_per_second": 16 / statistics.median(medians), "median_max_rss_bytes": statistics.median(block["arms"][arm]["max_aggregate_rss_bytes"] for block in status["blocks"]), "max_gpu_allocator_peak_bytes": max(block["arms"][arm]["max_gpu_allocator_peak_bytes"] for block in status["blocks"]), "median_cold_to_ready_seconds": statistics.median(block["arms"][arm]["max_worker_cold_to_ready_seconds"] for block in status["blocks"]), "median_total_arm_wall_seconds": statistics.median(block["arms"][arm]["wall_seconds_including_cold"] for block in status["blocks"]), "median_average_cpu_cores": statistics.median(statistics.median(row["average_cpu_cores_during_kernel"] for row in block["arms"][arm]["rounds"]) for block in status["blocks"])}
        comparisons = {}
        for candidate in ARMS:
            if candidate == "cpu_native_b16_xla":
                continue
            comparisons[f"{candidate}_over_cpu_native_b16_xla"] = paired_statistics(arm_summaries[candidate]["block_median_seconds"], arm_summaries["cpu_native_b16_xla"]["block_median_seconds"])
        comparisons["gpu_native_b16_xla_over_cpu_processes_16xb1_xla"] = paired_statistics(arm_summaries["gpu_native_b16_xla"]["block_median_seconds"], arm_summaries["cpu_processes_16xb1_xla"]["block_median_seconds"])
        status.update({"status": "complete", "finished_utc": _utc_now(), "summary": {"arms": arm_summaries, "paired_comparisons": comparisons, "interpretation": "matched_work_fresh_blocks_with_paired_uncertainty"}, "run_manifest": {"git_commit": _git_commit(), "command": [str(PYTHON), str(SCRIPT_PATH)], "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "UNSET"), "cpu_gpu_status": "cpu_hidden_per_cpu_worker_and_trusted_gpu0_growth_xla", "random_seed": RANDOM_SEED, "output_root": str(output_root.relative_to(REPO_ROOT)), "plan": PLAN, "result": RESULT, "trust_basis": GPU_TRUST_BASIS}, "nonclaims": ["no universal CPU/GPU/XLA superiority", "no equal-cost hardware comparison", "no larger-T extrapolation", "no default, HMC, posterior, production, or scientific-readiness claim"]})
        _write_json(output_root / "status.json", status)
        _write_json(output_root / "summary.json", status["summary"])
        return status
    except BaseException as exc:
        status.update({"status": "failed", "finished_utc": _utc_now(), "failure": {"type": type(exc).__name__, "message": str(exc)}})
        _write_json(output_root / "status.json", status)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--intra", type=int, default=1)
    parser.add_argument("--row-ids", default="")
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.worker:
        return _worker_main(args)
    if args.self_check:
        checks = {"arms": set(ARMS) == set(balanced_orders()[0]), "orders": all(set(order) == set(ARMS) for order in balanced_orders()), "rows": all(sorted(row for spec in arm_worker_specs(arm) for row in spec["row_ids"]) == list(ROW_IDS) for arm in ARMS), "topology": topology_contract()["valid"]}
        print(json.dumps(checks, indent=2, sort_keys=True))
        return 0 if all(checks.values()) else 1
    output_root = args.output_root.resolve()
    output_root.relative_to(REPO_ROOT)
    payload = run(output_root)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
