#!/usr/bin/env python3
"""Run bounded target-query multistart mode discovery for SSL-LSTM q=20."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN = Path("docs/plans/bayesfilter-ssl-lstm-q20-gap-closure-plan-2026-08-18.md")
RUNNER = Path("docs/benchmarks/run_ssl_lstm_q20_gap_closure_mode_discovery_2026_08_18.py")
GEOMETRY = Path(
    "docs/plans/artifacts/ssl-lstm-q20-seed-b-neutra-mode-failure-root-cause-2026-08-10/"
    "r1/geometry.json"
)
OUTPUT_ROOT = Path(
    "docs/plans/artifacts/ssl-lstm-q20-gap-closure-2026-08-18/discovery/r1"
)
PROGRESS = OUTPUT_ROOT / "progress.json"
FINAL = OUTPUT_ROOT / "discovery.json"

TARGET_SIGNATURE = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
GEOMETRY_SHA256 = "dc3dd7b84566867bc49c11ad16f50778d21457adbb398a17c2a75f3c3b461eeb"
PRIOR_CENTER = (0.35, -0.08, 0.65, 0.05)
PRIOR_SD = 4.0
RANDOM_SEED = 20260818
RANDOM_STARTS = 16
CORNER_STARTS = 16
WORKERS = 8
CORES_PER_WORKER = 4
CPU_START = 64
STATIONARY_SCORE_TOLERANCE = 1.0e-5
KNOWN_MODE_DISTANCE_TOLERANCE = 1.0e-3
COMPETING_LOG_PROB_GAP = 20.0
MAX_ITERATIONS = 200
WORKER_TIMEOUT_SECONDS = 3600.0


class DiscoveryError(RuntimeError):
    """Raised when the discovery diagnostic cannot produce valid evidence."""


def _abs(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha(path: Path) -> str:
    return hashlib.sha256(_abs(path).read_bytes()).hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "numpy"):
        return _safe(value.numpy())
    if hasattr(value, "tolist"):
        return _safe(value.tolist())
    if hasattr(value, "item"):
        return _safe(value.item())
    return value


def _write_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool = False) -> None:
    absolute = _abs(path)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if absolute.exists() and not overwrite:
        raise DiscoveryError(f"refusing to overwrite artifact: {path}")
    encoded = json.dumps(_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = absolute.with_suffix(absolute.suffix + ".tmp")
    temporary.write_text(encoded, encoding="ascii")
    temporary.replace(absolute)


def build_starts() -> tuple[Mapping[str, Any], ...]:
    rng = random.Random(RANDOM_SEED)
    rows: list[Mapping[str, Any]] = []
    for index in range(RANDOM_STARTS):
        rows.append(
            {
                "family": "prior_gaussian",
                "family_index": index,
                "position": tuple(
                    center + PRIOR_SD * rng.gauss(0.0, 1.0)
                    for center in PRIOR_CENTER
                ),
            }
        )
    for mask in range(CORNER_STARTS):
        rows.append(
            {
                "family": "prior_scale_hypercube_corner",
                "family_index": mask,
                "position": tuple(
                    center + PRIOR_SD * (1.0 if (mask >> axis) & 1 else -1.0)
                    for axis, center in enumerate(PRIOR_CENTER)
                ),
            }
        )
    return tuple(rows)


def classify_stationary_endpoints(
    rows: Sequence[Mapping[str, Any]],
    known_modes: Mapping[str, Sequence[float]],
) -> Mapping[str, Any]:
    stationary = [
        row
        for row in rows
        if math.isfinite(float(row["log_prob"]))
        and math.isfinite(float(row["score_inf_norm"]))
        and float(row["score_inf_norm"]) <= STATIONARY_SCORE_TOLERANCE
    ]
    best_known = max(float(row["log_prob"]) for row in rows if row.get("known_reference"))
    classified = []
    for row in stationary:
        position = tuple(float(value) for value in row["position"])
        distances = {
            label: max(abs(left - float(right)) for left, right in zip(position, center))
            for label, center in known_modes.items()
        }
        nearest_label = min(distances, key=distances.get)
        gap = best_known - float(row["log_prob"])
        classified.append(
            {
                **dict(row),
                "distance_inf_to_known": distances,
                "nearest_known_mode": nearest_label,
                "matches_known_mode": distances[nearest_label] <= KNOWN_MODE_DISTANCE_TOLERANCE,
                "log_prob_gap_to_best_known": gap,
                "new_competing_cluster_trigger": (
                    distances[nearest_label] > KNOWN_MODE_DISTANCE_TOLERANCE
                    and gap <= COMPETING_LOG_PROB_GAP
                ),
            }
        )
    triggers = [row for row in classified if row["new_competing_cluster_trigger"]]
    return {
        "stationary_count": len(stationary),
        "classified_stationary": classified,
        "new_competing_cluster_count": len(triggers),
        "new_competing_cluster_triggered": bool(triggers),
        "trigger_rows": triggers,
    }


def _worker(batch: Mapping[str, Any]) -> Mapping[str, Any]:
    cpu_ids = tuple(int(value) for value in batch["cpu_ids"])
    os.sched_setaffinity(0, set(cpu_ids))
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TF_NUM_INTRAOP_THREADS",
    ):
        os.environ[name] = str(CORES_PER_WORKER)
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"

    import tensorflow as tf
    import tensorflow_probability as tfp

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(CORES_PER_WORKER)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    if tf.config.list_physical_devices("GPU"):
        raise DiscoveryError("discovery worker found a visible GPU")

    from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import (
        complexity_posterior_target,
    )

    target = complexity_posterior_target(20, jit_compile=True)
    if target.target_signature() != TARGET_SIGNATURE:
        raise DiscoveryError("discovery worker target signature mismatch")

    def objective(position: Any) -> tuple[Any, Any]:
        value, score = target.log_prob_and_grad(tf.reshape(position, [4]))
        return -tf.convert_to_tensor(value, tf.float64), -tf.convert_to_tensor(score, tf.float64)

    compiled = tf.function(
        objective,
        input_signature=(tf.TensorSpec([4], tf.float64),),
        jit_compile=True,
        reduce_retracing=False,
    )
    output = []
    for item in batch["starts"]:
        started = time.perf_counter()
        result = tfp.optimizer.lbfgs_minimize(
            compiled,
            initial_position=tf.constant(item["position"], tf.float64),
            max_iterations=MAX_ITERATIONS,
            tolerance=tf.constant(1.0e-10, tf.float64),
            parallel_iterations=1,
        )
        position = tf.reshape(tf.convert_to_tensor(result.position, tf.float64), [4])
        value, score = target.log_prob_and_grad(position)
        output.append(
            {
                **dict(item),
                "position": position,
                "log_prob": value,
                "score_inf_norm": tf.reduce_max(tf.abs(score)),
                "converged": result.converged,
                "failed": result.failed,
                "iterations": result.num_iterations,
                "seconds": time.perf_counter() - started,
            }
        )
    return {
        "worker_index": int(batch["worker_index"]),
        "pid": os.getpid(),
        "configured_cpu_ids": cpu_ids,
        "actual_cpu_ids": tuple(sorted(os.sched_getaffinity(0))),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tensorflow_gpu_devices": tuple(str(value) for value in tf.config.list_physical_devices("GPU")),
        "target_signature": target.target_signature(),
        "jit_compile": True,
        "rows": output,
    }


def run() -> Mapping[str, Any]:
    started = time.perf_counter()
    if _abs(FINAL).exists():
        raise DiscoveryError("refusing to overwrite discovery result")
    if _sha(GEOMETRY) != GEOMETRY_SHA256:
        raise DiscoveryError("geometry artifact identity mismatch")
    geometry = json.loads(_abs(GEOMETRY).read_text(encoding="utf-8"))
    known_modes = {
        label: tuple(float(value) for value in geometry["representatives"][label]["position"])
        for label in ("plus", "minus")
    }
    starts = build_starts()
    batches = []
    for worker_index in range(WORKERS):
        batches.append(
            {
                "worker_index": worker_index,
                "cpu_ids": tuple(
                    range(
                        CPU_START + worker_index * CORES_PER_WORKER,
                        CPU_START + (worker_index + 1) * CORES_PER_WORKER,
                    )
                ),
                "starts": starts[worker_index::WORKERS],
            }
        )
    _write_json(
        PROGRESS,
        {"status": "DISCOVERY_STARTING", "completed_workers": 0, "total_workers": WORKERS},
        overwrite=True,
    )
    context = multiprocessing.get_context("spawn")
    worker_results = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=WORKERS, mp_context=context
    ) as executor:
        futures = [executor.submit(_worker, batch) for batch in batches]
        for future in concurrent.futures.as_completed(futures):
            worker_results.append(future.result(timeout=WORKER_TIMEOUT_SECONDS))
            _write_json(
                PROGRESS,
                {
                    "status": "DISCOVERY_RUNNING",
                    "completed_workers": len(worker_results),
                    "total_workers": WORKERS,
                    "elapsed_seconds": time.perf_counter() - started,
                },
                overwrite=True,
            )
    rows = [row for result in worker_results for row in result["rows"]]
    known_rows = []
    for label, position in known_modes.items():
        known_rows.append(
            {
                "family": "known_reference",
                "family_index": label,
                "position": position,
                "log_prob": geometry["representatives"][label]["log_prob"],
                "score_inf_norm": geometry["representatives"][label]["score_inf_norm"],
                "known_reference": True,
            }
        )
    classification = classify_stationary_endpoints(tuple(known_rows) + tuple(rows), known_modes)
    worker_identity_passed = all(
        result["configured_cpu_ids"] == result["actual_cpu_ids"]
        and result["cuda_visible_devices"] == "-1"
        and result["tensorflow_gpu_devices"] == ()
        and result["target_signature"] == TARGET_SIGNATURE
        and result["jit_compile"] is True
        for result in worker_results
    )
    stationary_new_rows = [
        row
        for row in classification["classified_stationary"]
        if row.get("family") != "known_reference"
    ]
    payload = {
        "schema": "bayesfilter.ssl_lstm.q20_gap_closure_mode_discovery.v1",
        "status": (
            "NEW_COMPETING_CLUSTER_TRIGGERED"
            if classification["new_competing_cluster_triggered"]
            else "NO_NEW_COMPETING_CLUSTER_FOUND_IN_BOUNDED_SEARCH"
        ),
        "question": "do broader target-query multistarts find a stationary cluster beyond the two known sign regions",
        "start_design": {
            "random_seed": RANDOM_SEED,
            "prior_center": PRIOR_CENTER,
            "prior_sd": PRIOR_SD,
            "random_prior_starts": RANDOM_STARTS,
            "prior_scale_hypercube_corners": CORNER_STARTS,
            "total_starts": len(starts),
        },
        "thresholds": {
            "stationary_score_inf_max": STATIONARY_SCORE_TOLERANCE,
            "known_mode_distance_inf_max": KNOWN_MODE_DISTANCE_TOLERANCE,
            "competing_log_prob_gap_max": COMPETING_LOG_PROB_GAP,
        },
        "hard_gates": {
            "worker_identity_passed": worker_identity_passed,
            "all_starts_returned": len(rows) == len(starts),
            "at_least_one_new_stationary_endpoint": bool(stationary_new_rows),
        },
        "classification": classification,
        "worker_results": worker_results,
        "bindings": {"geometry_sha256": _sha(GEOMETRY)},
        "run_manifest": {
            "git_commit": subprocess.run(
                ("git", "rev-parse", "HEAD"), cwd=ROOT, check=True,
                capture_output=True, text=True,
            ).stdout.strip(),
            "command": " ".join(sys.argv),
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
            "python": platform.python_version(),
            "cpu_gpu_status": "CPU_ONLY_GPU_HIDDEN",
            "jit_compile": True,
            "workers": WORKERS,
            "cores_per_worker": CORES_PER_WORKER,
            "cpu_ids": tuple(range(CPU_START, CPU_START + WORKERS * CORES_PER_WORKER)),
            "wall_seconds": time.perf_counter() - started,
            "plan_file": PLAN.as_posix(),
            "artifact_root": OUTPUT_ROOT.as_posix(),
            "source_sha256": {"runner": _sha(RUNNER), "plan": _sha(PLAN)},
        },
        "decision": (
            "enrich_global_proposal_before_posterior_claim"
            if classification["new_competing_cluster_triggered"]
            else "retain_explicit_nonexhaustive_two_region_limitation"
        ),
        "nonclaims": (
            "bounded multistart diagnostic cannot prove absence of additional modes",
            "local optimizer endpoints do not estimate posterior mode weights",
            "no sampler, posterior, predictive, superiority, or default claim",
        ),
    }
    if not worker_identity_passed or len(rows) != len(starts):
        payload["status"] = "DISCOVERY_HARD_GATE_FAILED"
    _write_json(FINAL, payload)
    _write_json(
        PROGRESS,
        {
            "status": payload["status"],
            "completed_workers": WORKERS,
            "total_workers": WORKERS,
            "result": FINAL.as_posix(),
            "elapsed_seconds": time.perf_counter() - started,
        },
        overwrite=True,
    )
    return payload


def main() -> None:
    payload = run()
    print(json.dumps({"status": payload["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
