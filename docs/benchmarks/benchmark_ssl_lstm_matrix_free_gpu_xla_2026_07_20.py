#!/usr/bin/env python3
"""Benchmark dense versus matrix-free SSL-LSTM selected UKF scores."""

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
SCRIPT = Path(__file__).resolve().relative_to(ROOT)
PLAN = Path("docs/plans/bayesfilter-ssl-lstm-matrix-free-filter-derivatives-plan-2026-07-20.md")
RESULT = Path("docs/plans/bayesfilter-ssl-lstm-matrix-free-filter-derivatives-result-2026-07-20.md")
DEFAULT_OUTPUT = Path(
    "docs/plans/artifacts/ssl-lstm-matrix-free-filter-derivatives-2026-07-20/"
    "gpu-xla-benchmark"
)
SCHEMA = "bayesfilter.ssl_lstm.matrix_free_gpu_xla_benchmark.v1"
WORKER_SCHEMA = "bayesfilter.ssl_lstm.matrix_free_gpu_xla_worker.v1"
TRUST_BASIS = "owner_designated_managed_session_visible_gpu_trusted"
Q_VALUES = (5, 10, 20)
ARMS = ("dense", "jvp")
REPETITIONS = 3
WARM_POINTS = 5
TIMEOUT_SECONDS = 1_200
HOST_RSS_CAP_BYTES = 64 * 1024**3
GPU_ALLOCATOR_CAP_BYTES = 28 * 1024**3
PARITY_TOLERANCE = 1.0e-10
NOMINATION_RATIO = 0.80
SMALL_Q_REGRESSION_RATIO = 1.10
MEMORY_RATIO_LIMIT = 1.10
MAX_PRELAUNCH_UTILIZATION = 50
SOURCE_PATHS = (
    SCRIPT,
    PLAN,
    Path("bayesfilter/nonlinear/svd_sigma_point_derivatives_tf.py"),
    Path("bayesfilter/nonlinear/ssl_lstm_sgqf_ukf_adapters.py"),
    Path("bayesfilter/nonlinear/ssl_lstm_complexity_target_tf.py"),
)


class BenchmarkError(RuntimeError):
    pass


def canonical(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest() -> dict[str, Any]:
    files = [
        {"path": path.as_posix(), "sha256": sha256(ROOT / path)}
        for path in SOURCE_PATHS
    ]
    return {
        "files": files,
        "fingerprint": hashlib.sha256(canonical(files)).hexdigest(),
    }


def repo_path(path: Path, *, label: str) -> Path:
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise BenchmarkError(f"{label} must remain inside the repository")
    return resolved


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical(payload))
    temporary.replace(path)


def strict_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BenchmarkError(f"expected JSON object: {path}")
    return payload


def planned_cells() -> list[tuple[int, int, str]]:
    cells: list[tuple[int, int, str]] = []
    for q in Q_VALUES:
        for repetition in range(REPETITIONS):
            order = ARMS if repetition % 2 == 0 else tuple(reversed(ARMS))
            cells.extend((q, repetition, arm) for arm in order)
    return cells


def _gpu_rows() -> list[dict[str, Any]]:
    completed = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ),
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    rows = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            continue
        rows.append(
            {
                "index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "memory_total_mib": int(fields[3]),
                "memory_used_mib": int(fields[4]),
                "utilization_percent": int(fields[5]),
            }
        )
    return rows


def _compute_apps() -> list[dict[str, Any]]:
    completed = subprocess.run(
        (
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    rows = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4 or not fields[1].isdigit():
            continue
        rows.append(
            {
                "gpu_uuid": fields[0],
                "pid": int(fields[1]),
                "process_name": fields[2],
                "used_memory_mib": int(fields[3]),
            }
        )
    return rows


def select_physical_gpu(requested: str) -> tuple[int, dict[str, Any]]:
    rows = _gpu_rows()
    apps = _compute_apps()
    by_index = {row["index"]: row for row in rows}
    if requested != "auto":
        selected = int(requested)
        if selected not in by_index:
            raise BenchmarkError(f"requested physical GPU {selected} is unavailable")
    else:
        selected = None
        for candidate in (1, 0):
            row = by_index.get(candidate)
            if row is None:
                continue
            foreign = [app for app in apps if app["gpu_uuid"] == row["uuid"]]
            if candidate == 1 and foreign:
                continue
            selected = candidate
            break
        if selected is None:
            raise BenchmarkError("neither physical GPU 1 nor GPU 0 is available")
    row = by_index[selected]
    selected_apps = [app for app in apps if app["gpu_uuid"] == row["uuid"]]
    return selected, {"gpu": row, "compute_apps": selected_apps}


def prelaunch_probe(
    physical_gpu: int,
    baseline_pids: set[int],
) -> dict[str, Any]:
    rows = _gpu_rows()
    row = next((item for item in rows if item["index"] == physical_gpu), None)
    if row is None:
        raise BenchmarkError("selected physical GPU disappeared")
    apps = [app for app in _compute_apps() if app["gpu_uuid"] == row["uuid"]]
    current_pids = {int(app["pid"]) for app in apps}
    new_pids = sorted(current_pids - baseline_pids)
    reasons = []
    if row["utilization_percent"] > MAX_PRELAUNCH_UTILIZATION:
        reasons.append("prelaunch_gpu_utilization_above_50_percent")
    if new_pids:
        reasons.append("new_foreign_compute_pid")
    return {
        "gpu": row,
        "compute_apps": apps,
        "baseline_compute_pids": sorted(baseline_pids),
        "new_compute_pids": new_pids,
        "timing_contaminated": bool(reasons),
        "contamination_reasons": reasons,
    }


def _proc_memory() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            name, amount, unit = line.split()[:3]
            if unit != "kB":
                raise BenchmarkError(f"unexpected /proc memory unit: {unit}")
            values[name.rstrip(":").lower() + "_bytes"] = int(amount) * 1024
    return values


def _worker_points(tf: Any, center: Any) -> Any:
    offsets = tf.constant(
        (
            (0.00, 0.00, 0.00, 0.00),
            (0.07, -0.05, 0.04, -0.03),
            (-0.06, 0.04, -0.02, 0.05),
            (0.11, 0.03, -0.08, -0.04),
            (-0.09, -0.02, 0.06, 0.07),
        ),
        dtype=tf.float64,
    )
    return center[tf.newaxis, :] + offsets


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false")
    started = time.perf_counter()

    import resource
    from dataclasses import replace

    import tensorflow as tf

    physical = tf.config.list_physical_devices("GPU")
    if len(physical) != 1:
        raise BenchmarkError(f"worker requires one visible physical GPU, found {len(physical)}")
    tf.config.experimental.set_memory_growth(physical[0], True)
    if tf.config.experimental.get_memory_growth(physical[0]) is not True:
        raise BenchmarkError("TensorFlow GPU memory growth was not established")
    tf.config.experimental.enable_tensor_float_32_execution(True)
    tf.config.set_soft_device_placement(False)
    logical = tf.config.list_logical_devices("GPU")
    if len(logical) != 1:
        raise BenchmarkError(f"worker requires one logical GPU, found {len(logical)}")
    tf.config.experimental.reset_memory_stats("GPU:0")

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import PRIOR_CENTER
    from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import complexity_posterior_target
    from bayesfilter.nonlinear.ssl_lstm_sgqf_ukf_adapters import make_ssl_lstm_svd_ukf_components
    from bayesfilter.nonlinear.svd_sigma_point_derivatives_tf import tf_principal_sqrt_ukf_score

    target = complexity_posterior_target(args.q, jit_compile=False)

    @tf.function(input_signature=[tf.TensorSpec([4], tf.float64)], jit_compile=True)
    def value_and_score(free):
        full = target.full_theta(free)
        components = make_ssl_lstm_svd_ukf_components(
            full,
            target.config.static_config,
            evidence_path=PLAN.as_posix(),
            derivative_parameter_indices=target.config.free_indices,
        )
        derivatives = components.derivatives
        if args.arm == "dense":
            derivatives = replace(
                derivatives,
                transition_jvp_fn=None,
                observation_jvp_fn=None,
            )
        result = tf_principal_sqrt_ukf_score(
            target.config.observations,
            components.model,
            derivatives,
            innovation_floor=tf.constant(1.0e-12, tf.float64),
        )
        delta = free - target.config.prior_center
        variance = tf.constant(target.config.prior_standard_deviation**2, tf.float64)
        value = result.log_likelihood - 0.5 * tf.reduce_sum(tf.square(delta) / variance)
        score = result.score - delta / variance
        return tf.ensure_shape(value, []), tf.ensure_shape(score, [4])

    points = _worker_points(tf, PRIOR_CENTER)
    first_started = time.perf_counter()
    first_value, first_score = value_and_score(points[0])
    first_value_value = float(first_value.numpy())
    first_score_value = [float(item) for item in first_score.numpy()]
    first_seconds = time.perf_counter() - first_started
    first_allocator = {
        key + "_bytes": int(value)
        for key, value in tf.config.experimental.get_memory_info("GPU:0").items()
    }
    tf.config.experimental.reset_memory_stats("GPU:0")

    warm_rows = []
    for index, point in enumerate(tf.unstack(points, axis=0)):
        call_started = time.perf_counter()
        value, score = value_and_score(point)
        value_value = float(value.numpy())
        score_value = [float(item) for item in score.numpy()]
        warm_rows.append(
            {
                "point_index": index,
                "seconds": time.perf_counter() - call_started,
                "value": value_value,
                "score": score_value,
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
        "target_signature": target.target_signature(),
        "first_call": {
            "seconds": first_seconds,
            "value": first_value_value,
            "score": first_score_value,
            "allocator": first_allocator,
        },
        "warm_rows": warm_rows,
        "warm_seconds_median": statistics.median(row["seconds"] for row in warm_rows),
        "warm_seconds_max": max(row["seconds"] for row in warm_rows),
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
            "physical_gpus": [device.name for device in physical],
            "logical_gpus": [device.name for device in logical],
            "jit_compile": True,
            "dtype": "float64",
            "tf32": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
            "memory_growth": True,
            "tf_force_gpu_allow_growth": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH"),
            "xla_flags": os.environ.get("XLA_FLAGS"),
            "wall_seconds": time.perf_counter() - started,
            "trust_basis": TRUST_BASIS,
            "plan": PLAN.as_posix(),
            "result": RESULT.as_posix(),
            "output": args.output.as_posix(),
        },
        "nonclaims": [
            "full selected-score engineering benchmark only",
            "no HMC, NeuTra, posterior, q=20 admission, or superiority claim",
        ],
    }
    write_json(repo_path(args.output, label="worker output"), payload)
    return payload


def validate_worker(payload: Mapping[str, Any], *, q: int, repetition: int, arm: str) -> None:
    if payload.get("schema") != WORKER_SCHEMA:
        raise BenchmarkError("worker schema mismatch")
    if (int(payload.get("q", -1)), int(payload.get("repetition", -1)), payload.get("arm")) != (
        q,
        repetition,
        arm,
    ):
        raise BenchmarkError("worker identity mismatch")
    if payload.get("status") != "PASSED" or payload.get("hard_vetoes") != []:
        raise BenchmarkError(f"worker hard veto at q={q}, repetition={repetition}, arm={arm}")
    manifest = payload.get("run_manifest", {})
    if manifest.get("jit_compile") is not True or manifest.get("dtype") != "float64":
        raise BenchmarkError("worker execution contract mismatch")
    if manifest.get("memory_growth") is not True or manifest.get("trust_basis") != TRUST_BASIS:
        raise BenchmarkError("worker trust/memory-growth contract mismatch")


def _parity(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float]:
    value_errors = []
    score_errors = []
    for dense_row, jvp_row in zip(left["warm_rows"], right["warm_rows"], strict=True):
        value_errors.append(abs(float(dense_row["value"]) - float(jvp_row["value"])))
        score_errors.extend(
            abs(float(a) - float(b))
            for a, b in zip(dense_row["score"], jvp_row["score"], strict=True)
        )
    return {
        "value_max_abs": max(value_errors),
        "score_max_abs": max(score_errors),
    }


def summarize_cells(cells: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    hard_vetoes = []
    for q in Q_VALUES:
        pairs = []
        for repetition in range(REPETITIONS):
            by_arm = {
                row["arm"]: row
                for row in cells
                if row["q"] == q and row["repetition"] == repetition
            }
            if set(by_arm) != set(ARMS):
                hard_vetoes.append(f"missing_pair_q{q}_r{repetition}")
                continue
            dense = by_arm["dense"]["worker"]
            jvp = by_arm["jvp"]["worker"]
            parity = _parity(dense, jvp)
            if max(parity.values()) > PARITY_TOLERANCE:
                hard_vetoes.append(f"parity_failure_q{q}_r{repetition}")
            dense_seconds = float(dense["warm_seconds_median"])
            jvp_seconds = float(jvp["warm_seconds_median"])
            pairs.append(
                {
                    "repetition": repetition,
                    "dense_seconds": dense_seconds,
                    "jvp_seconds": jvp_seconds,
                    "jvp_over_dense": jvp_seconds / dense_seconds,
                    "dense_allocator_peak_bytes": int(dense["gpu_allocator_memory"]["peak_bytes"]),
                    "jvp_allocator_peak_bytes": int(jvp["gpu_allocator_memory"]["peak_bytes"]),
                    "dense_host_hwm_bytes": max(
                        int(dense["process_memory"].get("vmhwm_bytes", 0)),
                        int(dense["process_memory"]["ru_maxrss_bytes"]),
                    ),
                    "jvp_host_hwm_bytes": max(
                        int(jvp["process_memory"].get("vmhwm_bytes", 0)),
                        int(jvp["process_memory"]["ru_maxrss_bytes"]),
                    ),
                    "parity": parity,
                    "timing_contaminated": bool(
                        by_arm["dense"]["prelaunch"]["timing_contaminated"]
                        or by_arm["jvp"]["prelaunch"]["timing_contaminated"]
                    ),
                }
            )
        ratios = [pair["jvp_over_dense"] for pair in pairs]
        allocator_ratios = [
            pair["jvp_allocator_peak_bytes"] / max(pair["dense_allocator_peak_bytes"], 1)
            for pair in pairs
        ]
        host_ratios = [
            pair["jvp_host_hwm_bytes"] / max(pair["dense_host_hwm_bytes"], 1)
            for pair in pairs
        ]
        rows.append(
            {
                "q": q,
                "pairs": pairs,
                "median_jvp_over_dense": statistics.median(ratios),
                "median_allocator_peak_ratio": statistics.median(allocator_ratios),
                "median_host_hwm_ratio": statistics.median(host_ratios),
                "any_timing_contaminated": any(pair["timing_contaminated"] for pair in pairs),
                "small_q_regression_signal": q < 20 and statistics.median(ratios) > SMALL_Q_REGRESSION_RATIO,
            }
        )
    q20 = next((row for row in rows if row["q"] == 20), None)
    nominate = bool(
        q20
        and not hard_vetoes
        and not q20["any_timing_contaminated"]
        and q20["median_jvp_over_dense"] <= NOMINATION_RATIO
        and q20["median_allocator_peak_ratio"] <= MEMORY_RATIO_LIMIT
        and q20["median_host_hwm_ratio"] <= MEMORY_RATIO_LIMIT
    )
    return {
        "rows": rows,
        "hard_vetoes": hard_vetoes,
        "downstream_q20_rerun_nominated": nominate,
        "decision": (
            "NOMINATE_BOUNDED_Q20_TARGET_NEUTRA_CAPACITY_RERUN"
            if nominate
            else "DO_NOT_NOMINATE_DOWNSTREAM_RERUN_FROM_CURRENT_EVIDENCE"
        ),
    }


def run_supervisor(args: argparse.Namespace) -> dict[str, Any]:
    output_root = repo_path(args.output_root, label="output root")
    summary_path = output_root / "summary.json"
    manifest = source_manifest()
    selected_gpu, initial_probe = select_physical_gpu(args.physical_gpu)
    baseline_pids = {int(app["pid"]) for app in initial_probe["compute_apps"]}
    started = time.perf_counter()
    cells = []
    for q, repetition, arm in planned_cells():
        if source_manifest()["fingerprint"] != manifest["fingerprint"]:
            raise BenchmarkError("source drift during benchmark")
        probe = prelaunch_probe(selected_gpu, baseline_pids)
        relative = args.output_root / f"q{q}-r{repetition}-{arm}.json"
        output = repo_path(relative, label="cell output")
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
        cell_started = time.perf_counter()
        log.parent.mkdir(parents=True, exist_ok=True)
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
            raise BenchmarkError(
                f"worker failed at q={q}, repetition={repetition}, arm={arm}; log={log}"
            )
        worker = strict_json(output)
        validate_worker(worker, q=q, repetition=repetition, arm=arm)
        cells.append(
            {
                "q": q,
                "repetition": repetition,
                "arm": arm,
                "command": list(command),
                "prelaunch": probe,
                "worker": worker,
                "output": relative.as_posix(),
                "log": log.relative_to(ROOT).as_posix(),
                "supervisor_wall_seconds": time.perf_counter() - cell_started,
            }
        )
        partial = {
            "schema": SCHEMA,
            "status": "RUNNING",
            "selected_physical_gpu": selected_gpu,
            "source_manifest": manifest,
            "initial_gpu_probe": initial_probe,
            "cells": cells,
        }
        write_json(summary_path, partial)
    summary = summarize_cells(cells)
    payload = {
        "schema": SCHEMA,
        "status": "PASSED" if not summary["hard_vetoes"] else "HARD_VETO",
        "selected_physical_gpu": selected_gpu,
        "source_manifest": manifest,
        "initial_gpu_probe": initial_probe,
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
            "dtype": "float64",
            "paired_repetitions": REPETITIONS,
            "warm_points_per_worker": WARM_POINTS,
            "trust_basis": TRUST_BASIS,
            "plan": PLAN.as_posix(),
            "result": RESULT.as_posix(),
            "output": summary_path.relative_to(ROOT).as_posix(),
        },
        "inference_status": {
            "hard_veto_screen": "passed" if not summary["hard_vetoes"] else "failed",
            "statistically_supported_ranking": "none",
            "descriptive_only_differences": [
                "paired fresh-process warm wall ratios",
                "TensorFlow allocator current/peak bytes",
                "process VmHWM and ru_maxrss",
            ],
            "default_readiness": "not_assessed",
            "next_evidence_needed": (
                "bounded q20 downstream capacity rerun"
                if summary["downstream_q20_rerun_nominated"]
                else "repair or additional uncontaminated performance evidence"
            ),
        },
        "nonclaims": [
            "three repetitions provide descriptive engineering evidence only",
            "no statistical superiority, q20 admission, NeuTra, HMC, posterior, or default claim",
        ],
    }
    write_json(summary_path, payload)
    return payload


def contract_payload() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "mode": "contract-smoke",
        "q_values": list(Q_VALUES),
        "arms": list(ARMS),
        "repetitions": REPETITIONS,
        "warm_points": WARM_POINTS,
        "cell_count": len(planned_cells()),
        "parity_tolerance": PARITY_TOLERANCE,
        "nomination_ratio": NOMINATION_RATIO,
        "memory_ratio_limit": MEMORY_RATIO_LIMIT,
        "timeout_seconds": TIMEOUT_SECONDS,
        "host_rss_cap_bytes": HOST_RSS_CAP_BYTES,
        "gpu_allocator_cap_bytes": GPU_ALLOCATOR_CAP_BYTES,
        "material_execution_authorized": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("contract-smoke", "worker", "supervisor"), required=True)
    parser.add_argument("--q", type=int, choices=Q_VALUES)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--repetition", type=int)
    parser.add_argument("--physical-gpu", default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--authorize-gpu-benchmark", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "worker" and any(
        value is None for value in (args.q, args.arm, args.repetition, args.output)
    ):
        parser.error("worker mode requires q, arm, repetition, and output")
    if args.mode == "supervisor" and not args.authorize_gpu_benchmark:
        parser.error("supervisor mode requires --authorize-gpu-benchmark")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "contract-smoke":
        payload = contract_payload()
    elif args.mode == "worker":
        payload = run_worker(args)
    else:
        payload = run_supervisor(args)
    print(
        "JSON_SUMMARY "
        + json.dumps(
            {
                "mode": args.mode,
                "status": payload.get("status", "CONTRACT"),
                "decision": payload.get("summary", {}).get("decision"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
