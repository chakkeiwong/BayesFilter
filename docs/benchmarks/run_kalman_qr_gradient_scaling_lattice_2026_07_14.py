#!/usr/bin/env python
"""Run the repaired Kalman QR gradient scaling lattice sequentially."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import signal
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/home/ubuntu/anaconda3/envs/tfgpu/bin/python")
RUNNER = REPO_ROOT / "docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py"
PLAN = "docs/plans/bayesfilter-kalman-qr-gradient-scaling-lattice-gpu0-plan-2026-07-14.md"
RESULT = "docs/plans/bayesfilter-kalman-qr-gradient-scaling-lattice-gpu0-result-2026-07-14.md"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "docs/benchmarks/kalman_qr_gradient_scaling_lattice_gpu0_r1_2026-07-14"
)
DIMENSIONS = (10, 20, 30)
PARAMETER_COUNTS = (50, 150)
BATCH_SIZES = (1, 4, 16)
CPU_THREADS = (1, 4, 16)
GPU_DTYPES = ("float32", "float64")
METHODS = (
    "batch_native_analytical_qr_score",
    "batch_native_autodiff_qr_score",
)
MEASUREMENT_SOURCE_PATHS = (
    "bayesfilter/linear/kalman_qr_tf.py",
    "bayesfilter/linear/kalman_qr_derivatives_tf.py",
    "bayesfilter/linear/qr_factor_tf.py",
    "scripts/kalman_qr_benchmark_contract.py",
    "scripts/benchmark_kalman_qr_parameter_count_scaling.py",
    "docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py",
)
CPU_INHERITANCE_SOURCE_PATHS = (
    "bayesfilter/linear/kalman_qr_tf.py",
    "bayesfilter/linear/kalman_qr_derivatives_tf.py",
    "bayesfilter/linear/qr_factor_tf.py",
    "scripts/kalman_qr_benchmark_contract.py",
)
CPU_INHERITANCE_GPU_ONLY_DRIFT_PATHS = (
    "scripts/benchmark_kalman_qr_parameter_count_scaling.py",
    "docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py",
)
CONTROL_SOURCE_PATHS = (
    "docs/benchmarks/run_kalman_qr_gradient_scaling_lattice_2026_07_14.py",
    PLAN,
)
SOURCE_PATHS = MEASUREMENT_SOURCE_PATHS + CONTROL_SOURCE_PATHS
SCHEMA = "bayesfilter.kalman_qr.gradient_scaling_lattice.v1"
SUMMARY_SCHEMA = "bayesfilter.kalman_qr.gradient_scaling_lattice.summary.v1"
GPU_TRUST_BASIS = "owner_designated_managed_session_visible_gpu_trusted"
GPU_XLA_FLAG = "--xla_gpu_enable_triton_gemm=false"
MAX_LOAD_ONE_MINUTE = 64.0
MAX_FOREIGN_CPU_PERCENT = 1600.0
GPU_PHYSICAL_INDEX = 0
GPU_ALLOWED_DISPLAY_PIDS = (5955, 6575)
MAX_GPU_SHARED_BASELINE_MEMORY_MIB = 2048
MAX_GPU_PRELAUNCH_UTILIZATION_PERCENT = 50


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _source_manifest() -> dict[str, Any]:
    files = []
    for relative in SOURCE_PATHS:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"missing execution source: {relative}")
        files.append({"path": relative, "sha256": _sha256(path)})
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"files": files, "fingerprint": hashlib.sha256(encoded).hexdigest()}


def _manifest_hashes(
    manifest: Mapping[str, Any], paths: Sequence[str]
) -> dict[str, str]:
    files = manifest.get("files")
    if not isinstance(files, list):
        raise RuntimeError("source manifest has no file list")
    by_path: dict[str, str] = {}
    for row in files:
        if not isinstance(row, Mapping):
            raise RuntimeError("source manifest contains a malformed row")
        path = row.get("path")
        digest = row.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str) or not digest:
            raise RuntimeError("source manifest contains an invalid path or hash")
        if path in by_path:
            raise RuntimeError(f"source manifest contains duplicate path: {path}")
        by_path[path] = digest
    if any(path not in by_path for path in paths):
        raise RuntimeError("source manifest is missing a measurement source")
    return {path: by_path[path] for path in paths}


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "UNKNOWN"


def _foreign_compute_processes() -> list[dict[str, Any]]:
    completed = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,stat=,pcpu=,nlwp=,args="],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if completed.returncode != 0:
        return [{"pid": None, "command": "process census failed"}]
    parsed = []
    for line in completed.stdout.splitlines():
        fields = line.strip().split(None, 5)
        if len(fields) == 6:
            parsed.append(fields)
    descendants = {os.getpid()}
    changed = True
    while changed:
        changed = False
        for pid, ppid, *_rest in parsed:
            if int(ppid) in descendants and int(pid) not in descendants:
                descendants.add(int(pid))
                changed = True
    relevant = (
        "run_kalman_qr",
        "benchmark_kalman_qr",
        "run_ssl_lstm_predictive_validation",
        "verify_ssl_lstm_predictive_validation",
        "/home/ubuntu/anaconda3/envs/tfgpu/bin/python",
    )
    rows = []
    for fields in parsed:
        pid, ppid, state, cpu_percent, threads, command = fields
        if int(pid) in descendants:
            continue
        if any(token in command for token in relevant):
            rows.append(
                {
                    "pid": int(pid),
                    "ppid": int(ppid),
                    "state": state,
                    "cpu_percent": float(cpu_percent),
                    "threads": int(threads),
                    "command": command,
                }
            )
    return rows


def _gpu_snapshot() -> dict[str, Any]:
    gpu_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    process_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return {
        "gpu_query_returncode": gpu_query.returncode,
        "gpu_rows": gpu_query.stdout.splitlines(),
        "process_query_returncode": process_query.returncode,
        "compute_process_rows": process_query.stdout.splitlines(),
    }


def _gpu_target_identity(snapshot: Mapping[str, Any]) -> tuple[str, int, int] | None:
    if snapshot.get("gpu_query_returncode") != 0:
        return None
    rows = snapshot.get("gpu_rows")
    if not isinstance(rows, list) or len(rows) < 2:
        return None
    matching = [
        [field.strip() for field in str(row).split(",")]
        for row in rows
        if str(row).split(",", 1)[0].strip() == str(GPU_PHYSICAL_INDEX)
    ]
    if len(matching) != 1:
        return None
    fields = matching[0]
    if len(fields) != 7:
        return None
    try:
        return fields[1], int(fields[4]), int(fields[5])
    except ValueError:
        return None


def _gpu_target_pids(snapshot: Mapping[str, Any], gpu_uuid: str) -> list[int] | None:
    processes = snapshot.get("compute_process_rows")
    if snapshot.get("process_query_returncode") != 0 or not isinstance(
        processes, list
    ):
        return None
    pids = []
    for row in processes:
        fields = [field.strip() for field in str(row).split(",")]
        if not fields or fields[0] != gpu_uuid:
            continue
        if len(fields) < 2:
            return None
        try:
            pids.append(int(fields[1]))
        except ValueError:
            return None
    return pids


def _gpu_target_prelaunch_admissible(snapshot: Mapping[str, Any]) -> bool:
    identity = _gpu_target_identity(snapshot)
    if identity is None:
        return False
    gpu_uuid, memory_used, utilization = identity
    pids = _gpu_target_pids(snapshot, gpu_uuid)
    if pids is None:
        return False
    return (
        memory_used <= MAX_GPU_SHARED_BASELINE_MEMORY_MIB
        and utilization < MAX_GPU_PRELAUNCH_UTILIZATION_PERCENT
        and set(pids) == set(GPU_ALLOWED_DISPLAY_PIDS)
    )


def _pid_is_in_process_group(pid: int, pgid: int) -> bool:
    try:
        return os.getpgid(pid) == pgid
    except (OSError, ProcessLookupError):
        return False


def _gpu_target_runtime_admissible(
    snapshot: Mapping[str, Any], *, owned_pgid: int
) -> bool:
    """Allow the display baseline and benchmark-owned GPU work only."""

    identity = _gpu_target_identity(snapshot)
    if identity is None:
        return False
    gpu_uuid, memory_used, _utilization = identity
    matching_pids = _gpu_target_pids(snapshot, gpu_uuid)
    if matching_pids is None or not set(GPU_ALLOWED_DISPLAY_PIDS).issubset(
        matching_pids
    ):
        return False
    owned_seen = False
    for pid in matching_pids:
        if pid in GPU_ALLOWED_DISPLAY_PIDS:
            continue
        if not _pid_is_in_process_group(pid, owned_pgid):
            return False
        owned_seen = True
    return owned_seen or memory_used <= MAX_GPU_SHARED_BASELINE_MEMORY_MIB


def _gpu_target_released(snapshot: Mapping[str, Any]) -> bool:
    """Require the benchmark GPU context to be gone after its runner exits."""

    identity = _gpu_target_identity(snapshot)
    if identity is None:
        return False
    gpu_uuid, memory_used, _utilization = identity
    pids = _gpu_target_pids(snapshot, gpu_uuid)
    if pids is None:
        return False
    return (
        memory_used <= MAX_GPU_SHARED_BASELINE_MEMORY_MIB
        and set(pids) == set(GPU_ALLOWED_DISPLAY_PIDS)
    )


def _resource_snapshot(device: str) -> dict[str, Any]:
    return {
        "observed_utc": _utc_now(),
        "load_average": list(os.getloadavg()),
        "foreign_compute_processes": _foreign_compute_processes(),
        "gpu": _gpu_snapshot() if device == "gpu" else None,
    }


def _host_resources_idle(snapshot: Mapping[str, Any], device: str) -> bool:
    foreign = snapshot.get("foreign_compute_processes", [])
    if not isinstance(foreign, list):
        return False
    foreign_cpu_percent = 0.0
    for row in foreign:
        if not isinstance(row, Mapping) or row.get("pid") is None:
            return False
        try:
            foreign_cpu_percent += float(row["cpu_percent"])
        except (KeyError, TypeError, ValueError):
            return False
    load_average = snapshot.get("load_average", [])
    if (
        not isinstance(load_average, list)
        or not load_average
        or float(load_average[0]) > MAX_LOAD_ONE_MINUTE
    ):
        return False
    return device == "gpu" or foreign_cpu_percent <= MAX_FOREIGN_CPU_PERCENT


def _resources_idle(snapshot: Mapping[str, Any], device: str) -> bool:
    """Prelaunch admission: bounded host load and authorized shared GPU 0."""

    if not _host_resources_idle(snapshot, device):
        return False
    return device != "gpu" or _gpu_target_prelaunch_admissible(snapshot["gpu"])


def _runtime_resources_idle(
    snapshot: Mapping[str, Any], device: str, *, owned_pgid: int
) -> bool:
    """Runtime admission: bounded host load and no unapproved GPU 0 process."""

    if not _host_resources_idle(snapshot, device):
        return False
    return device != "gpu" or _gpu_target_runtime_admissible(
        snapshot["gpu"], owned_pgid=owned_pgid
    )


def _post_run_resources_released(snapshot: Mapping[str, Any], device: str) -> bool:
    if not _host_resources_idle(snapshot, device):
        return False
    return device != "gpu" or _gpu_target_released(snapshot["gpu"])


def _schedule_specs() -> list[dict[str, Any]]:
    specs = []
    for threads in CPU_THREADS:
        for batch_size in BATCH_SIZES:
            specs.append(
                {
                    "schedule_id": f"cpu-t{threads}-b{batch_size}-float32",
                    "device": "cpu",
                    "dtype": "float32",
                    "batch_size": batch_size,
                    "cpu_threads": threads,
                    "tf32_enabled": False,
                }
            )
    for dtype in GPU_DTYPES:
        for batch_size in BATCH_SIZES:
            specs.append(
                {
                    "schedule_id": f"gpu-b{batch_size}-{dtype}",
                    "device": "gpu",
                    "dtype": dtype,
                    "batch_size": batch_size,
                    "cpu_threads": 1,
                    "tf32_enabled": dtype == "float32",
                }
            )
    return specs


def _command(spec: Mapping[str, Any], output_dir: Path, timeout_seconds: int) -> list[str]:
    return [
        str(PYTHON),
        str(RUNNER),
        "--dimensions",
        *(str(value) for value in DIMENSIONS),
        "--parameter-counts",
        *(str(value) for value in PARAMETER_COUNTS),
        "--timesteps",
        "120",
        "--batch-size",
        str(spec["batch_size"]),
        "--dtype",
        str(spec["dtype"]),
        "--device",
        str(spec["device"]),
        "--cpu-threads",
        str(spec["cpu_threads"]),
        "--repeats",
        "5",
        "--timeout-seconds",
        str(timeout_seconds),
        "--methods",
        *METHODS,
        "--output-dir",
        str(output_dir),
        "--plan-path",
        PLAN,
        "--result-path",
        RESULT,
        "--no-resume",
        "--jit-compile",
        "--tf32-enabled" if spec["tf32_enabled"] else "--no-tf32",
    ]


def _status_valid(payload: Mapping[str, Any] | None, spec: Mapping[str, Any]) -> bool:
    if not payload or payload.get("status") != "complete":
        return False
    if payload.get("comparison_summary", {}).get("comparison_complete") is not True:
        return False
    checks = payload.get("aggregate_checks")
    records = payload.get("records")
    if not isinstance(checks, Mapping) or not checks or not all(checks.values()):
        return False
    if not isinstance(records, list) or len(records) != 12:
        return False
    for record in records:
        measurement = record.get("measurement")
        device = record.get("device_manifest")
        if (
            record.get("state") != "passed"
            or not isinstance(measurement, Mapping)
            or len(measurement.get("durations", {}).get("warm_execution_seconds", [])) != 5
            or not isinstance(device, Mapping)
        ):
            return False
        if spec["device"] == "gpu":
            policy = record.get("gpu_xla_triton_gemm_policy")
            growth = record.get("gpu_memory_growth_policy")
            allocator = record.get("gpu_allocator_memory")
            if (
                device.get("selected_device") != "/GPU:0"
                or device.get("trust_basis") != GPU_TRUST_BASIS
                or not isinstance(policy, Mapping)
                or policy.get("action") != "benchmark_default_no_triton_applied"
                or policy.get("input_xla_flags") != "UNSET"
                or policy.get("effective_xla_flags") != GPU_XLA_FLAG
                or record.get("xla_flags") != GPU_XLA_FLAG
                or not isinstance(growth, Mapping)
                or growth.get("policy") != "required_no_full_device_preallocation"
                or growth.get("environment_variable") != "TF_FORCE_GPU_ALLOW_GROWTH"
                or growth.get("environment_value") != "true"
                or not isinstance(allocator, Mapping)
                or allocator.get("device") != "/GPU:0"
                or type(allocator.get("current_bytes")) is not int
                or allocator["current_bytes"] < 0
                or type(allocator.get("peak_bytes")) is not int
                or allocator["peak_bytes"] < allocator["current_bytes"]
            ):
                return False
        elif (
            device.get("selected_device") != "/CPU:0"
            or record.get("xla_flags") != "UNSET"
        ):
            return False
    return True


def _run_spec(
    spec: Mapping[str, Any],
    *,
    output_root: Path,
    timeout_seconds: int,
    attempt: int,
) -> dict[str, Any]:
    schedule_id = str(spec["schedule_id"])
    output_dir = output_root / schedule_id / f"attempt-{attempt}"
    log_path = output_root / "logs" / f"{schedule_id}-attempt-{attempt}.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = _command(spec, output_dir, timeout_seconds)
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["OMP_NUM_THREADS"] = str(spec["cpu_threads"])
    environment["TF_NUM_INTRAOP_THREADS"] = str(spec["cpu_threads"])
    environment["TF_NUM_INTEROP_THREADS"] = str(spec["cpu_threads"])
    environment["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    environment.pop("XLA_FLAGS", None)
    environment["CUDA_VISIBLE_DEVICES"] = (
        "-1" if spec["device"] == "cpu" else str(GPU_PHYSICAL_INDEX)
    )
    resource_before = _resource_snapshot(str(spec["device"]))
    started = time.perf_counter()
    started_utc = _utc_now()
    overlap_samples = []
    runtime_resource_samples = []
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while True:
                try:
                    returncode = process.wait(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    snapshot = _resource_snapshot(str(spec["device"]))
                    runtime_resource_samples.append(snapshot)
                    if not _runtime_resources_idle(
                        snapshot,
                        str(spec["device"]),
                        owned_pgid=process.pid,
                    ):
                        overlap_samples.append(snapshot)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            raise
    elapsed = time.perf_counter() - started
    resource_after = _resource_snapshot(str(spec["device"]))
    status_path = output_dir / "status.json"
    status = _read_json(status_path)
    overlap = bool(overlap_samples) or not _post_run_resources_released(
        resource_after, str(spec["device"])
    )
    structured_valid = _status_valid(status, spec)
    return {
        "attempt": attempt,
        "command": command,
        "device_environment": {
            "CUDA_VISIBLE_DEVICES": environment["CUDA_VISIBLE_DEVICES"],
            "OMP_NUM_THREADS": environment["OMP_NUM_THREADS"],
            "TF_NUM_INTRAOP_THREADS": environment["TF_NUM_INTRAOP_THREADS"],
            "TF_NUM_INTEROP_THREADS": environment["TF_NUM_INTEROP_THREADS"],
            "XLA_FLAGS": environment.get("XLA_FLAGS", "UNSET"),
            "TF_FORCE_GPU_ALLOW_GROWTH": environment["TF_FORCE_GPU_ALLOW_GROWTH"],
        },
        "elapsed_seconds": elapsed,
        "finished_utc": _utc_now(),
        "log_path": str(log_path.relative_to(REPO_ROOT)),
        "output_dir": str(output_dir.relative_to(REPO_ROOT)),
        "overlap_samples": overlap_samples,
        "overlap_veto": overlap,
        "runtime_owned_process_group": process.pid,
        "runtime_resource_samples": runtime_resource_samples,
        "resource_after": resource_after,
        "resource_before": resource_before,
        "returncode": returncode,
        "started_utc": started_utc,
        "status_path": str(status_path.relative_to(REPO_ROOT)),
        "structured_status_valid": structured_valid,
        "valid": returncode == 0 and structured_valid and not overlap,
    }


def _record_warm_rows(
    spec: Mapping[str, Any], status: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    for record in status.get("records", []):
        durations = record["measurement"]["durations"]
        warm = [float(value) for value in durations["warm_execution_seconds"]]
        rows.append(
            {
                "batch_size": spec["batch_size"],
                "cpu_threads": spec["cpu_threads"] if spec["device"] == "cpu" else None,
                "device": spec["device"],
                "dimension": int(
                    record["case_id"].split("dimension=", 1)[1].split("-", 1)[0]
                ),
                "dtype": spec["dtype"],
                "first_executable_call_seconds": durations[
                    "first_executable_call_seconds"
                ],
                "graphdef_nodes": record["measurement"]["graphdef"]["node_count"],
                "graphdef_serialized_bytes": record["measurement"]["graphdef"][
                    "serialized_bytes"
                ],
                "method_id": record["method_id"],
                "parameter_count": int(
                    record["case_id"].split("parameter_count=", 1)[1].split("-", 1)[0]
                ),
                "trace_seconds": durations["trace_seconds"],
                "warm_execution_seconds": warm,
                "warm_mean_seconds": statistics.fmean(warm),
                "warm_median_seconds": statistics.median(warm),
                "warm_min_seconds": min(warm),
                "warm_max_seconds": max(warm),
                "gpu_allocator_current_bytes": (
                    record.get("gpu_allocator_memory", {}).get("current_bytes")
                    if spec["device"] == "gpu"
                    else None
                ),
                "gpu_allocator_peak_bytes": (
                    record.get("gpu_allocator_memory", {}).get("peak_bytes")
                    if spec["device"] == "gpu"
                    else None
                ),
                "gpu_memory_growth": (
                    record.get("gpu_memory_growth_policy", {}).get(
                        "environment_value"
                    )
                    == "true"
                    if spec["device"] == "gpu"
                    else None
                ),
            }
        )
    return rows


def _write_summary(output_root: Path, master: Mapping[str, Any]) -> None:
    rows = []
    for schedule in master.get("schedules", []):
        if schedule.get("state") != "passed":
            continue
        attempt = schedule["attempts"][-1]
        status_path = Path(attempt["status_path"])
        if not status_path.is_absolute():
            status_path = REPO_ROOT / status_path
        inheritance = schedule.get("inheritance")
        if isinstance(inheritance, Mapping) and (
            not status_path.is_file()
            or inheritance.get("status_sha256") != _sha256(status_path)
        ):
            raise RuntimeError("inherited CPU schedule status drifted")
        status = _read_json(status_path)
        if status is not None:
            rows.extend(_record_warm_rows(schedule["spec"], status))
    _write_json(
        output_root / "summary.json",
        {
            "schema": SUMMARY_SCHEMA,
            "generated_utc": _utc_now(),
            "row_count": len(rows),
            "rows": rows,
            "interpretation": "descriptive_only_no_statistically_supported_ranking",
        },
    )


def _relative_repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError as exc:
        raise ValueError(f"path must remain under repository root: {path}") from exc


def _execution_contract(
    output_root: Path, inherited_cpu_status: Path | None = None
) -> dict[str, Any]:
    return {
        "dimensions": list(DIMENSIONS),
        "parameter_counts": list(PARAMETER_COUNTS),
        "timesteps": 120,
        "batch_sizes": list(BATCH_SIZES),
        "cpu_thread_limits": list(CPU_THREADS),
        "gpu_dtypes": list(GPU_DTYPES),
        "methods": list(METHODS),
        "warm_repeats_per_method_child": 5,
        "jit_compile": True,
        "gpu_physical_index": GPU_PHYSICAL_INDEX,
        "gpu_trust_basis": GPU_TRUST_BASIS,
        "resource_policy": {
            "max_load_one_minute": MAX_LOAD_ONE_MINUTE,
            "cpu_schedule_max_foreign_cpu_percent": MAX_FOREIGN_CPU_PERCENT,
            "gpu_schedule_foreign_cpu_role": (
                "explanatory only; one-minute host load remains a veto"
            ),
            "gpu_prelaunch": (
                "physical GPU 0 has exactly the authorized display PIDs, at most "
                "2048 MiB used, and less than 50 percent utilization"
            ),
            "gpu_runtime": (
                "physical GPU 0 compute PIDs must be the authorized display PIDs "
                "or belong to the benchmark-owned process group"
            ),
            "gpu_allowed_display_pids": list(GPU_ALLOWED_DISPLAY_PIDS),
            "gpu_memory_growth": "required and validated per method record",
        },
        "plan": PLAN,
        "result": RESULT,
        "output_root": _relative_repo_path(output_root),
        "inherited_cpu_status": (
            _relative_repo_path(inherited_cpu_status)
            if inherited_cpu_status is not None
            else None
        ),
    }


def _inheritance_contract_core() -> dict[str, Any]:
    return {
        "dimensions": list(DIMENSIONS),
        "parameter_counts": list(PARAMETER_COUNTS),
        "timesteps": 120,
        "batch_sizes": list(BATCH_SIZES),
        "cpu_thread_limits": list(CPU_THREADS),
        "methods": list(METHODS),
        "warm_repeats_per_method_child": 5,
        "jit_compile": True,
    }


def _validated_inherited_cpu_schedules(
    source_status_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prior = _read_json(source_status_path)
    if prior is None or prior.get("schema") != SCHEMA:
        raise RuntimeError("CPU inheritance source is missing or has the wrong schema")
    if prior.get("status") not in ("complete", "complete_with_failures"):
        raise RuntimeError("CPU inheritance source is not terminal")
    prior_core = prior.get("execution_contract")
    if not isinstance(prior_core, Mapping):
        raise RuntimeError("CPU inheritance source has no execution contract")
    for key, expected in _inheritance_contract_core().items():
        if prior_core.get(key) != expected:
            raise RuntimeError(f"CPU inheritance contract mismatch: {key}")

    current_manifest = _source_manifest()
    current_hashes = _manifest_hashes(
        current_manifest, CPU_INHERITANCE_SOURCE_PATHS
    )
    prior_hashes = _manifest_hashes(
        prior.get("source_manifest", {}), CPU_INHERITANCE_SOURCE_PATHS
    )
    if prior_hashes != current_hashes:
        raise RuntimeError("CPU inheritance mathematical/contract sources have drifted")
    prior_gpu_only_hashes = _manifest_hashes(
        prior.get("source_manifest", {}), CPU_INHERITANCE_GPU_ONLY_DRIFT_PATHS
    )
    current_gpu_only_hashes = _manifest_hashes(
        current_manifest, CPU_INHERITANCE_GPU_ONLY_DRIFT_PATHS
    )

    schedules = prior.get("schedules")
    if not isinstance(schedules, list):
        raise RuntimeError("CPU inheritance source has no schedules")
    by_id: dict[str, Mapping[str, Any]] = {}
    for schedule in schedules:
        if not isinstance(schedule, Mapping):
            continue
        schedule_id = str(schedule.get("spec", {}).get("schedule_id", ""))
        if schedule_id in by_id:
            raise RuntimeError(f"duplicate inherited schedule: {schedule_id}")
        by_id[schedule_id] = schedule

    inherited = []
    for spec in (row for row in _schedule_specs() if row["device"] == "cpu"):
        schedule_id = str(spec["schedule_id"])
        schedule = by_id.get(schedule_id)
        if schedule is None or schedule.get("state") != "passed":
            raise RuntimeError(f"CPU inheritance schedule is not passed: {schedule_id}")
        if schedule.get("spec") != spec:
            raise RuntimeError(f"CPU inheritance spec mismatch: {schedule_id}")
        attempts = schedule.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            raise RuntimeError(f"CPU inheritance has no attempt: {schedule_id}")
        accepted = attempts[-1]
        if (
            not isinstance(accepted, Mapping)
            or accepted.get("valid") is not True
            or accepted.get("structured_status_valid") is not True
            or accepted.get("overlap_veto") is not False
            or accepted.get("returncode") != 0
        ):
            raise RuntimeError(f"CPU inheritance attempt is not admissible: {schedule_id}")
        environment = accepted.get("device_environment")
        if (
            not isinstance(environment, Mapping)
            or environment.get("CUDA_VISIBLE_DEVICES") != "-1"
            or environment.get("OMP_NUM_THREADS") != str(spec["cpu_threads"])
            or environment.get("TF_NUM_INTRAOP_THREADS")
            != str(spec["cpu_threads"])
            or environment.get("TF_NUM_INTEROP_THREADS")
            != str(spec["cpu_threads"])
            or environment.get("XLA_FLAGS") != "UNSET"
        ):
            raise RuntimeError(f"CPU inheritance environment mismatch: {schedule_id}")
        status_path_value = accepted.get("status_path")
        if not isinstance(status_path_value, str):
            raise RuntimeError(f"CPU inheritance status path missing: {schedule_id}")
        status_path = Path(status_path_value)
        if not status_path.is_absolute():
            status_path = REPO_ROOT / status_path
        _relative_repo_path(status_path)
        if not _status_valid(_read_json(status_path), spec):
            raise RuntimeError(f"CPU inheritance structured status invalid: {schedule_id}")
        status_sha256 = _sha256(status_path)
        inherited.append(
            {
                "spec": copy.deepcopy(spec),
                "state": "passed",
                "attempts": [copy.deepcopy(dict(accepted))],
                "inheritance": {
                    "source_status": _relative_repo_path(source_status_path),
                    "accepted_attempt": accepted.get("attempt"),
                    "status_sha256": status_sha256,
                    "validated_utc": _utc_now(),
                },
            }
        )
    return inherited, {
        "source_status": _relative_repo_path(source_status_path),
        "source_status_sha256": _sha256(source_status_path),
        "cpu_schedule_count": len(inherited),
        "method_record_count": (
            len(inherited)
            * len(DIMENSIONS)
            * len(PARAMETER_COUNTS)
            * len(METHODS)
        ),
        "cpu_inheritance_source_hashes": current_hashes,
        "gpu_only_harness_drift": {
            path: {
                "prior_sha256": prior_gpu_only_hashes[path],
                "current_sha256": current_gpu_only_hashes[path],
                "classification": "gpu_memory_growth_or_gpu_admission_only",
            }
            for path in CPU_INHERITANCE_GPU_ONLY_DRIFT_PATHS
        },
        "validation": (
            "structured CPU statuses/environments and unchanged mathematical/"
            "contract sources revalidated; GPU-only harness drift excluded"
        ),
    }


def _inherited_artifacts_valid(master: Mapping[str, Any]) -> bool:
    provenance = master.get("inheritance")
    if provenance is None:
        return True
    if not isinstance(provenance, Mapping):
        return False
    source_status_value = provenance.get("source_status")
    if not isinstance(source_status_value, str):
        return False
    source_status = REPO_ROOT / source_status_value
    if (
        not source_status.is_file()
        or provenance.get("source_status_sha256") != _sha256(source_status)
    ):
        return False
    inherited_count = 0
    for schedule in master.get("schedules", []):
        if not isinstance(schedule, Mapping) or "inheritance" not in schedule:
            continue
        inherited_count += 1
        inheritance = schedule.get("inheritance")
        attempts = schedule.get("attempts")
        spec = schedule.get("spec")
        if (
            not isinstance(inheritance, Mapping)
            or not isinstance(attempts, list)
            or len(attempts) != 1
            or not isinstance(attempts[0], Mapping)
            or not isinstance(spec, Mapping)
        ):
            return False
        status_path_value = attempts[0].get("status_path")
        if not isinstance(status_path_value, str):
            return False
        status_path = Path(status_path_value)
        if not status_path.is_absolute():
            status_path = REPO_ROOT / status_path
        if (
            not status_path.is_file()
            or inheritance.get("status_sha256") != _sha256(status_path)
            or not _status_valid(_read_json(status_path), spec)
        ):
            return False
    return inherited_count == provenance.get("cpu_schedule_count") == 9


def _new_master(
    output_root: Path, inherited_cpu_status: Path | None = None
) -> dict[str, Any]:
    schedules: list[dict[str, Any]] = []
    inheritance = None
    if inherited_cpu_status is not None:
        schedules, inheritance = _validated_inherited_cpu_schedules(
            inherited_cpu_status
        )
    master = {
        "schema": SCHEMA,
        "started_utc": _utc_now(),
        "updated_utc": _utc_now(),
        "status": "running",
        "git_commit": _git_commit(),
        "source_manifest": _source_manifest(),
        "execution_contract": _execution_contract(output_root, inherited_cpu_status),
        "evidence_contract": {
            "primary_gate": (
                "all method records complete, finite, expected dtype/shape, "
                "and analytical/autodiff parity"
            ),
            "hard_vetoes": [
                "method timeout/crash/failure",
                "nonfinite or dtype/shape failure",
                "analytical/autodiff parity failure",
                "source drift",
                "CPU schedule foreign compute overlap after one retry",
                "GPU schedule host load above 64 after one retry",
                "GPU 0 outside the authorized shared-device prelaunch gate",
                "GPU memory-growth or allocator telemetry missing",
            ],
            "explanatory_only": [
                "first-call time",
                "warm-call times",
                "GraphDef size",
                "thread/dtype/batch scaling",
            ],
            "nonclaims": [
                "no statistically supported speed ranking",
                "no physical-core pinning claim",
                "no universal hardware or TensorFlow conclusion",
                "no HMC/posterior/default/production/scientific claim",
            ],
        },
        "schedules": schedules,
    }
    if inheritance is not None:
        master["inheritance"] = inheritance
    return master


def _load_or_create_master(
    output_root: Path, inherited_cpu_status: Path | None = None
) -> dict[str, Any]:
    path = output_root / "status.json"
    existing = _read_json(path)
    if existing is None:
        return _new_master(output_root, inherited_cpu_status)
    if (
        existing.get("schema") != SCHEMA
        or existing.get("execution_contract")
        != _execution_contract(output_root, inherited_cpu_status)
        or existing.get("source_manifest", {}).get("fingerprint")
        != _source_manifest()["fingerprint"]
    ):
        raise RuntimeError("existing lattice status does not match current execution contract")
    if not _inherited_artifacts_valid(existing):
        raise RuntimeError("existing lattice inherited CPU artifacts no longer validate")
    return existing


def _wait_for_resources(
    device: str,
    *,
    master: dict[str, Any],
    master_path: Path,
    wait_seconds: int,
    poll_seconds: int,
) -> bool:
    started = time.monotonic()
    while True:
        snapshot = _resource_snapshot(device)
        if _resources_idle(snapshot, device):
            master.pop("resource_wait", None)
            master["status"] = "running"
            master["updated_utc"] = _utc_now()
            _write_json(master_path, master)
            return True
        master["status"] = "waiting_for_resources"
        master["resource_wait"] = {"device": device, "snapshot": snapshot}
        master["updated_utc"] = _utc_now()
        _write_json(master_path, master)
        if time.monotonic() - started >= wait_seconds:
            return False
        time.sleep(poll_seconds)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--method-timeout-seconds", type=int, default=600)
    parser.add_argument("--resource-wait-seconds", type=int, default=7200)
    parser.add_argument("--resource-poll-seconds", type=int, default=30)
    parser.add_argument("--inherit-passed-cpu-from", type=Path)
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args(argv)


def _self_check() -> int:
    specs = _schedule_specs()
    checks = {
        "schedule_count": len(specs) == 15,
        "cpu_schedule_count": sum(spec["device"] == "cpu" for spec in specs) == 9,
        "gpu_schedule_count": sum(spec["device"] == "gpu" for spec in specs) == 6,
        "unique_ids": len({spec["schedule_id"] for spec in specs}) == 15,
        "expected_records_per_schedule": (
            len(DIMENSIONS) * len(PARAMETER_COUNTS) * len(METHODS) == 12
        ),
    }
    print(json.dumps(checks, indent=2, sort_keys=True))
    return 0 if all(checks.values()) else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.self_check:
        return _self_check()
    if min(
        args.method_timeout_seconds,
        args.resource_wait_seconds,
        args.resource_poll_seconds,
    ) <= 0:
        raise ValueError("timeouts and poll interval must be positive")
    output_root = args.output_root.resolve()
    _relative_repo_path(output_root)
    inherited_cpu_status = args.inherit_passed_cpu_from
    if inherited_cpu_status is not None:
        if not inherited_cpu_status.is_absolute():
            inherited_cpu_status = REPO_ROOT / inherited_cpu_status
        inherited_cpu_status = inherited_cpu_status.resolve()
        _relative_repo_path(inherited_cpu_status)
        if inherited_cpu_status == output_root / "status.json":
            raise ValueError("CPU inheritance source must be a prior result root")
    output_root.mkdir(parents=True, exist_ok=True)
    master_path = output_root / "status.json"
    master = _load_or_create_master(output_root, inherited_cpu_status)
    _write_json(master_path, master)
    opening_fingerprint = master["source_manifest"]["fingerprint"]
    by_id = {schedule["spec"]["schedule_id"]: schedule for schedule in master["schedules"]}
    failed = False
    for spec in _schedule_specs():
        schedule_id = spec["schedule_id"]
        schedule = by_id.get(schedule_id)
        if schedule is not None and schedule.get("state") == "passed":
            continue
        if schedule is None:
            schedule = {"spec": spec, "state": "pending", "attempts": []}
            master["schedules"].append(schedule)
            by_id[schedule_id] = schedule
        if schedule.get("state") == "running":
            schedule["state"] = "interrupted_prior_run"
        if schedule.get("state") == "failed" or schedule.get("state") == (
            "failed_overlap_after_retry"
        ):
            failed = True
            break
        for _ in range(2 - len(schedule["attempts"])):
            if not _wait_for_resources(
                spec["device"],
                master=master,
                master_path=master_path,
                wait_seconds=args.resource_wait_seconds,
                poll_seconds=args.resource_poll_seconds,
            ):
                master["status"] = "blocked_resource_not_exclusive"
                master["updated_utc"] = _utc_now()
                _write_json(master_path, master)
                _write_summary(output_root, master)
                return 2
            if _source_manifest()["fingerprint"] != opening_fingerprint:
                master["status"] = "failed_source_drift"
                master["updated_utc"] = _utc_now()
                _write_json(master_path, master)
                _write_summary(output_root, master)
                return 3
            if not _inherited_artifacts_valid(master):
                master["status"] = "failed_inherited_artifact_drift"
                master["updated_utc"] = _utc_now()
                _write_json(master_path, master)
                return 3
            schedule["state"] = "running"
            master["updated_utc"] = _utc_now()
            _write_json(master_path, master)
            result = _run_spec(
                spec,
                output_root=output_root,
                timeout_seconds=args.method_timeout_seconds,
                attempt=len(schedule["attempts"]) + 1,
            )
            schedule["attempts"].append(result)
            if _source_manifest()["fingerprint"] != opening_fingerprint:
                master["status"] = "failed_source_drift"
                master["updated_utc"] = _utc_now()
                _write_json(master_path, master)
                _write_summary(output_root, master)
                return 3
            if not _inherited_artifacts_valid(master):
                master["status"] = "failed_inherited_artifact_drift"
                master["updated_utc"] = _utc_now()
                _write_json(master_path, master)
                return 3
            if result["valid"]:
                schedule["state"] = "passed"
                break
            if not result["overlap_veto"]:
                schedule["state"] = "failed"
                failed = True
                break
            schedule["state"] = "overlap_retry_pending"
        if schedule.get("state") not in ("passed", "failed"):
            schedule["state"] = "failed_overlap_after_retry"
            failed = True
        master["updated_utc"] = _utc_now()
        _write_json(master_path, master)
        _write_summary(output_root, master)
        if failed:
            break
    master["status"] = "complete_with_failures" if failed else "complete"
    master["finished_utc"] = _utc_now()
    master["updated_utc"] = master["finished_utc"]
    _write_json(master_path, master)
    _write_summary(output_root, master)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
