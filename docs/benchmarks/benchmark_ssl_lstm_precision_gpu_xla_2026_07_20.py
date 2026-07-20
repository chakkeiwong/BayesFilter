#!/usr/bin/env python3
"""Trusted GPU/XLA accuracy and speed comparison for SSL-LSTM precision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path("docs/benchmarks/benchmark_ssl_lstm_precision_gpu_xla_2026_07_20.py")
PLAN = Path("docs/plans/bayesfilter-ssl-lstm-precision-accuracy-speed-plan-2026-07-20.md")
RESULT = Path("docs/plans/bayesfilter-ssl-lstm-precision-accuracy-speed-result-2026-07-20.md")
OUTPUT_ROOT = Path("docs/plans/artifacts/ssl-lstm-precision-accuracy-speed-2026-07-20/gpu-xla")
SCHEMA = "bayesfilter.ssl_lstm.precision_gpu_xla.v1"
WORKER_SCHEMA = "bayesfilter.ssl_lstm.precision_gpu_xla_worker.v1"
TRUST_BASIS = "owner_designated_managed_session_visible_gpu_trusted"
Q_VALUES = (5, 10, 20)
ARMS = ("all_float64", "mixed_lstm32_filter64", "all_float32_tf32")
REPETITIONS = 2
WARM_POINTS = 5
TIMEOUT_SECONDS = 600
MAX_PRELAUNCH_UTILIZATION = 50
GPU_ALLOCATOR_CAP_BYTES = 28 * 1024**3
HOST_RSS_CAP_BYTES = 64 * 1024**3
MIXED_LIMITS = {"value_abs": 2.0e-5, "score_abs": 2.0e-4, "score_scaled": 2.0e-4}
FLOAT32_LIMITS = {"value_abs": 2.0e-3, "score_abs": 2.0e-2, "score_scaled": 2.0e-3}
SOURCE_PATHS = (
    Path("bayesfilter/nonlinear/ssl_lstm_precision_experiment_tf.py"),
    Path("bayesfilter/nonlinear/ssl_lstm_complexity_target_tf.py"),
    Path("bayesfilter/nonlinear/ssl_lstm_sgqf_ukf_adapters.py"),
    SCRIPT,
    PLAN,
)


class BenchmarkError(RuntimeError):
    pass


def _canonical(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(payload))


def strict_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def source_manifest() -> dict[str, Any]:
    rows = []
    digest = hashlib.sha256()
    for relative in SOURCE_PATHS:
        data = (ROOT / relative).read_bytes()
        value = hashlib.sha256(data).hexdigest()
        rows.append({"path": relative.as_posix(), "sha256": value})
        digest.update(relative.as_posix().encode("ascii") + b"\0" + value.encode("ascii"))
    return {"files": rows, "fingerprint": digest.hexdigest()}


def _csv(command: tuple[str, ...]) -> list[list[str]]:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise BenchmarkError(completed.stderr.strip() or "nvidia-smi failed")
    return [
        [part.strip() for part in line.split(",")]
        for line in completed.stdout.splitlines()
        if line.strip()
    ]


def gpu_rows() -> list[dict[str, Any]]:
    rows = _csv(
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        )
    )
    return [
        {
            "index": int(row[0]),
            "uuid": row[1],
            "name": row[2],
            "utilization_percent": int(row[3]),
            "memory_used_mib": int(row[4]),
            "memory_total_mib": int(row[5]),
        }
        for row in rows
    ]


def compute_apps() -> list[dict[str, Any]]:
    rows = _csv(
        (
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        )
    )
    return [
        {
            "gpu_uuid": row[0],
            "pid": int(row[1]),
            "process_name": row[2],
            "used_memory_mib": int(row[3]),
        }
        for row in rows
    ]


def select_gpu(requested: int | None) -> tuple[int, dict[str, Any]]:
    by_index = {row["index"]: row for row in gpu_rows()}
    apps = compute_apps()
    order = (1, 0) if requested is None else (requested,)
    for index in order:
        row = by_index.get(index)
        if row is None:
            continue
        selected_apps = [app for app in apps if app["gpu_uuid"] == row["uuid"]]
        foreign_python = [app for app in selected_apps if "python" in app["process_name"].lower()]
        if not foreign_python:
            return index, {"gpu": row, "compute_apps": selected_apps}
    raise BenchmarkError("no requested GPU is available without a foreign Python process")


def prelaunch_probe(index: int, baseline_pids: set[int]) -> dict[str, Any]:
    row = next(item for item in gpu_rows() if item["index"] == index)
    apps = [app for app in compute_apps() if app["gpu_uuid"] == row["uuid"]]
    new_pids = sorted({app["pid"] for app in apps} - baseline_pids)
    reasons = []
    if row["utilization_percent"] > MAX_PRELAUNCH_UTILIZATION:
        reasons.append("prelaunch_gpu_utilization_above_50_percent")
    if new_pids:
        reasons.append("new_foreign_compute_pid")
    return {
        "gpu": row,
        "compute_apps": apps,
        "new_compute_pids": new_pids,
        "timing_contaminated": bool(reasons),
        "contamination_reasons": reasons,
    }


def planned_cells() -> list[tuple[int, int, str]]:
    rows = []
    for q in Q_VALUES:
        for repetition in range(REPETITIONS):
            order = ARMS if repetition % 2 == 0 else tuple(reversed(ARMS))
            rows.extend((q, repetition, arm) for arm in order)
    return rows


def _proc_memory() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            name, amount, unit = line.split()[:3]
            if unit != "kB":
                raise BenchmarkError("unexpected process memory unit")
            values[name.rstrip(":").lower() + "_bytes"] = int(amount) * 1024
    return values


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false")
    started = time.perf_counter()
    import resource

    import tensorflow as tf

    physical = tf.config.list_physical_devices("GPU")
    if len(physical) != 1:
        raise BenchmarkError(f"worker requires exactly one visible GPU, found {len(physical)}")
    tf.config.experimental.set_memory_growth(physical[0], True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    tf.config.set_soft_device_placement(False)
    logical = tf.config.list_logical_devices("GPU")
    if len(logical) != 1:
        raise BenchmarkError("worker requires one logical GPU")
    tf.config.experimental.reset_memory_stats("GPU:0")

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import PRIOR_CENTER
    from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import complexity_posterior_target
    from bayesfilter.nonlinear.ssl_lstm_precision_experiment_tf import policy_dtypes
    from bayesfilter.nonlinear.ssl_lstm_precision_experiment_tf import (
        ssl_lstm_precision_value_and_score,
    )

    target = complexity_posterior_target(args.q, jit_compile=False)
    _, filter_dtype = policy_dtypes(args.arm)

    @tf.function(
        input_signature=[tf.TensorSpec([4], filter_dtype)],
        jit_compile=True,
        reduce_retracing=True,
    )
    def value_and_score(free):
        result = ssl_lstm_precision_value_and_score(
            free,
            target.config.fixture,
            target.config.observations,
            target.config.static_config,
            target.config.free_indices,
            policy=args.arm,
            prior_center=target.config.prior_center,
            prior_standard_deviation=target.config.prior_standard_deviation,
        )
        return (
            result.value,
            result.score,
            result.placement_floor_count,
            result.innovation_floor_count,
            result.max_factor_reconstruction_residual,
        )

    offsets = tf.constant(
        (
            (0.00, 0.00, 0.00, 0.00),
            (0.07, -0.05, 0.04, -0.03),
            (-0.06, 0.04, -0.02, 0.05),
            (0.11, 0.03, -0.08, -0.04),
            (-0.09, -0.02, 0.06, 0.07),
        ),
        dtype=filter_dtype,
    )
    points = tf.cast(PRIOR_CENTER, filter_dtype)[tf.newaxis, :] + offsets
    first_started = time.perf_counter()
    first = value_and_score(points[0])
    [item.numpy() for item in first]
    first_seconds = time.perf_counter() - first_started
    first_allocator = {
        key + "_bytes": int(value)
        for key, value in tf.config.experimental.get_memory_info("GPU:0").items()
    }
    tf.config.experimental.reset_memory_stats("GPU:0")

    warm_rows = []
    for index, point in enumerate(tf.unstack(points, axis=0)):
        call_started = time.perf_counter()
        value, score, placement_floors, innovation_floors, residual = value_and_score(point)
        value_value = float(value.numpy())
        score_value = [float(item) for item in score.numpy()]
        warm_rows.append(
            {
                "point_index": index,
                "seconds": time.perf_counter() - call_started,
                "value": value_value,
                "score": score_value,
                "placement_floor_count": int(placement_floors.numpy()),
                "innovation_floor_count": int(innovation_floors.numpy()),
                "factor_reconstruction_residual": float(residual.numpy()),
                "finite": bool(
                    tf.math.is_finite(value).numpy()
                    and tf.reduce_all(tf.math.is_finite(score)).numpy()
                ),
                "devices": sorted({str(value.device), str(score.device)}),
            }
        )
    allocator = {
        key + "_bytes": int(value)
        for key, value in tf.config.experimental.get_memory_info("GPU:0").items()
    }
    proc_memory = _proc_memory()
    ru_maxrss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    hard_vetoes = []
    if not all(row["finite"] for row in warm_rows):
        hard_vetoes.append("nonfinite_output")
    if not all(any("GPU:0" in device for device in row["devices"]) for row in warm_rows):
        hard_vetoes.append("gpu_output_placement_missing")
    if allocator["peak_bytes"] > GPU_ALLOCATOR_CAP_BYTES:
        hard_vetoes.append("gpu_allocator_cap_exceeded")
    if max(proc_memory.get("vmhwm_bytes", 0), ru_maxrss_bytes) > HOST_RSS_CAP_BYTES:
        hard_vetoes.append("host_rss_cap_exceeded")
    payload = {
        "schema": WORKER_SCHEMA,
        "status": "PASSED" if not hard_vetoes else "HARD_VETO",
        "q": args.q,
        "arm": args.arm,
        "repetition": args.repetition,
        "hard_vetoes": hard_vetoes,
        "first_call_seconds": first_seconds,
        "first_allocator": first_allocator,
        "warm_rows": warm_rows,
        "warm_seconds_median": statistics.median(row["seconds"] for row in warm_rows),
        "gpu_allocator_memory": allocator,
        "process_memory": {**proc_memory, "ru_maxrss_bytes": ru_maxrss_bytes},
        "run_manifest": {
            "git_commit": subprocess.check_output(
                ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
            ).strip(),
            "command": " ".join(sys.argv),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "selected_physical_gpu": args.physical_gpu,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "logical_gpus": [device.name for device in logical],
            "jit_compile": True,
            "storage_dtype": filter_dtype.name,
            "model_dtype": policy_dtypes(args.arm)[0].name,
            "tf32_policy_enabled": bool(
                tf.config.experimental.tensor_float_32_execution_enabled()
            ),
            "memory_growth": True,
            "xla_flags": os.environ.get("XLA_FLAGS"),
            "wall_seconds": time.perf_counter() - started,
            "trust_basis": TRUST_BASIS,
            "plan": PLAN.as_posix(),
            "result": RESULT.as_posix(),
            "output": args.output.as_posix(),
        },
        "nonclaims": [
            "precision engineering comparison only",
            "no HMC, NeuTra, posterior, default, or superiority claim",
        ],
    }
    write_json(ROOT / args.output, payload)
    return payload


def _errors(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    value_errors = []
    score_errors = []
    scaled_errors = []
    branch_changes = []
    for left, right in zip(reference["warm_rows"], candidate["warm_rows"], strict=True):
        value_errors.append(abs(float(left["value"]) - float(right["value"])))
        for baseline, value in zip(left["score"], right["score"], strict=True):
            error = abs(float(baseline) - float(value))
            score_errors.append(error)
            scaled_errors.append(error / max(1.0, abs(float(baseline))))
        branch_changes.append(
            int(left["placement_floor_count"]) != int(right["placement_floor_count"])
            or int(left["innovation_floor_count"]) != int(right["innovation_floor_count"])
        )
    return {
        "value_max_abs": max(value_errors),
        "score_max_abs": max(score_errors),
        "score_max_scaled": max(scaled_errors),
        "floor_branch_changed": any(branch_changes),
    }


def summarize(cells: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    hard_vetoes = []
    for q in Q_VALUES:
        repetitions = []
        for repetition in range(REPETITIONS):
            by_arm = {
                row["arm"]: row
                for row in cells
                if row["q"] == q and row["repetition"] == repetition
            }
            if set(by_arm) != set(ARMS):
                hard_vetoes.append(f"missing_cells_q{q}_r{repetition}")
                continue
            reference = by_arm["all_float64"]["worker"]
            candidates = {}
            for arm in ARMS[1:]:
                worker = by_arm[arm]["worker"]
                candidates[arm] = {
                    "errors": _errors(reference, worker),
                    "warm_seconds": worker["warm_seconds_median"],
                    "speed_ratio_to_fp64": worker["warm_seconds_median"]
                    / reference["warm_seconds_median"],
                    "allocator_peak_bytes": worker["gpu_allocator_memory"]["peak_bytes"],
                }
            repetitions.append(
                {
                    "repetition": repetition,
                    "fp64_warm_seconds": reference["warm_seconds_median"],
                    "fp64_allocator_peak_bytes": reference["gpu_allocator_memory"]["peak_bytes"],
                    "candidates": candidates,
                    "timing_contaminated": any(
                        row["prelaunch"]["timing_contaminated"] for row in by_arm.values()
                    ),
                }
            )
        arm_summaries = {}
        for arm, limits in (
            ("mixed_lstm32_filter64", MIXED_LIMITS),
            ("all_float32_tf32", FLOAT32_LIMITS),
        ):
            errors = [row["candidates"][arm]["errors"] for row in repetitions]
            max_errors = {
                key: max(float(item[key]) for item in errors)
                for key in ("value_max_abs", "score_max_abs", "score_max_scaled")
            }
            branch_changed = any(bool(item["floor_branch_changed"]) for item in errors)
            accuracy_passed = bool(
                max_errors["value_max_abs"] <= limits["value_abs"]
                and max_errors["score_max_abs"] <= limits["score_abs"]
                and max_errors["score_max_scaled"] <= limits["score_scaled"]
                and not branch_changed
            )
            arm_summaries[arm] = {
                **max_errors,
                "floor_branch_changed": branch_changed,
                "accuracy_passed": accuracy_passed,
                "median_speed_ratio_to_fp64": statistics.median(
                    row["candidates"][arm]["speed_ratio_to_fp64"] for row in repetitions
                ),
                "any_timing_contaminated": any(
                    row["timing_contaminated"] for row in repetitions
                ),
            }
        rows.append({"q": q, "repetitions": repetitions, "arms": arm_summaries})
    return {"rows": rows, "hard_vetoes": hard_vetoes}


def run_supervisor(args: argparse.Namespace) -> dict[str, Any]:
    output_root = ROOT / args.output_root
    manifest = source_manifest()
    selected_gpu, initial_probe = select_gpu(args.physical_gpu)
    baseline_pids = {app["pid"] for app in initial_probe["compute_apps"]}
    cells = []
    started = time.perf_counter()
    for q, repetition, arm in planned_cells():
        if source_manifest()["fingerprint"] != manifest["fingerprint"]:
            raise BenchmarkError("source drift during benchmark")
        probe = prelaunch_probe(selected_gpu, baseline_pids)
        relative = args.output_root / f"q{q}-r{repetition}-{arm}.json"
        output = ROOT / relative
        log = output.with_suffix(".log")
        command = (
            sys.executable,
            str(ROOT / SCRIPT),
            "--mode",
            "worker",
            "--q",
            str(q),
            "--arm",
            arm,
            "--repetition",
            str(repetition),
            "--physical-gpu",
            str(selected_gpu),
            "--output",
            relative.as_posix(),
        )
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
        environment["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        environment.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false")
        log.parent.mkdir(parents=True, exist_ok=True)
        cell_started = time.perf_counter()
        with log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=TIMEOUT_SECONDS,
                check=False,
            )
        if completed.returncode != 0 or not output.is_file():
            raise BenchmarkError(f"worker failed: {relative}; see {log.relative_to(ROOT)}")
        worker = strict_json(output)
        if worker.get("status") != "PASSED":
            raise BenchmarkError(f"worker veto: {relative}")
        cells.append(
            {
                "q": q,
                "repetition": repetition,
                "arm": arm,
                "prelaunch": probe,
                "worker": worker,
                "output": relative.as_posix(),
                "log": log.relative_to(ROOT).as_posix(),
                "supervisor_wall_seconds": time.perf_counter() - cell_started,
            }
        )
        write_json(
            output_root / "summary.json",
            {
                "schema": SCHEMA,
                "status": "RUNNING",
                "selected_physical_gpu": selected_gpu,
                "source_manifest": manifest,
                "cells": cells,
            },
        )
    summary = summarize(cells)
    payload = {
        "schema": SCHEMA,
        "status": "PASSED" if not summary["hard_vetoes"] else "HARD_VETO",
        "selected_physical_gpu": selected_gpu,
        "initial_gpu_probe": initial_probe,
        "source_manifest": manifest,
        "cells": cells,
        "summary": summary,
        "run_manifest": {
            "git_commit": subprocess.check_output(
                ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
            ).strip(),
            "command": " ".join(sys.argv),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
            "python": sys.version.split()[0],
            "wall_seconds": time.perf_counter() - started,
            "selected_physical_gpu": selected_gpu,
            "jit_compile": True,
            "tf32_policy_enabled": True,
            "repetitions": REPETITIONS,
            "warm_points": WARM_POINTS,
            "trust_basis": TRUST_BASIS,
            "plan": PLAN.as_posix(),
            "result": RESULT.as_posix(),
            "output": (args.output_root / "summary.json").as_posix(),
        },
        "inference_status": {
            "hard_veto_screen": "reported per candidate",
            "statistically_supported_ranking": "none",
            "descriptive_only_differences": "two-repetition warm timing and memory",
            "default_readiness": "not_assessed",
            "next_evidence_needed": "candidate-specific decision after accuracy vetoes",
        },
        "nonclaims": [
            "no statistically supported speed ranking",
            "no HMC, NeuTra, posterior, scientific, or default claim",
        ],
    }
    write_json(output_root / "summary.json", payload)
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--mode", choices=("worker", "supervisor", "contract"), required=True)
    result.add_argument("--q", type=int, choices=Q_VALUES)
    result.add_argument("--arm", choices=ARMS)
    result.add_argument("--repetition", type=int, default=0)
    result.add_argument("--physical-gpu", type=int)
    result.add_argument("--output", type=Path)
    result.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    result.add_argument("--authorize-gpu-benchmark", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.mode == "contract":
        print(
            json.dumps(
                {
                    "q_values": list(Q_VALUES),
                    "arms": list(ARMS),
                    "repetitions": REPETITIONS,
                    "cell_count": len(planned_cells()),
                    "warm_points": WARM_POINTS,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.mode == "worker":
        if args.q is None or args.arm is None or args.output is None:
            raise BenchmarkError("worker requires q, arm, and output")
        run_worker(args)
        return 0
    if not args.authorize_gpu_benchmark:
        raise BenchmarkError("supervisor requires --authorize-gpu-benchmark")
    run_supervisor(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
