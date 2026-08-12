#!/usr/bin/env python3
"""Run bounded bridge-correct AIS diagnostics for SSL-LSTM q=20."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN = Path("docs/plans/bayesfilter-ssl-lstm-q20-physical-ais-repair-plan-2026-08-10.md")
RESULT = Path("docs/plans/bayesfilter-ssl-lstm-q20-physical-ais-repair-result-2026-08-10.md")
RUNNER = Path("docs/benchmarks/run_ssl_lstm_q20_physical_ais_repair_2026_08_10.py")
AIS_HELPER = Path("bayesfilter/testing/annealed_importance_tf.py")
IMPORTANCE_HELPER = Path("bayesfilter/testing/importance_sampling_tf.py")
GEOMETRY = Path("docs/plans/artifacts/ssl-lstm-q20-seed-b-neutra-mode-failure-root-cause-2026-08-10/r1/geometry.json")
DIRECT_WEIGHT = Path("docs/plans/artifacts/ssl-lstm-q20-physical-global-repair-2026-08-10/r1/weights.json")
OUTPUT_ROOT = Path("docs/plans/artifacts/ssl-lstm-q20-physical-ais-repair-2026-08-10/r1")
SPARSE_OUTPUT_ROOT = Path("docs/plans/artifacts/ssl-lstm-q20-physical-ais-repair-2026-08-10/r2")

TARGET_SIGNATURE = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
ADAPTER_SIGNATURE = "a8be6c212eb2a74ef926ccb9279871805949cae6e00c312a12525141638166f3"
GEOMETRY_SHA256 = "dc3dd7b84566867bc49c11ad16f50778d21457adbb398a17c2a75f3c3b461eeb"
DIRECT_WEIGHT_SHA256 = "ed683da0cdc95fbd51a8ec5bd25554f3c711b739736dfabf0d56c6ddf7d64075"
PARAMETER_DIM = 4
PATHS_PER_WORKER = 4
CORES_PER_WORKER = 4
CANARY_STEPS = 16
MATERIAL_STEPS = 64
SENSITIVITY_STEPS = 32
HMC_STEP_SIZE = 0.03
HMC_LEAPFROG = 4
FULL_REJUVENATION_INTERVAL = 1
SPARSE_REJUVENATION_INTERVAL = 8
CANARY_CAP_SECONDS = 1800.0
CAMPAIGN_CAP_SECONDS = 7200.0
MATERIAL_WORKERS = 25
CENTRAL_BATCHES = 8
SENSITIVITY_BATCHES = 2


class AISRepairError(RuntimeError):
    """Raised when an AIS evidence invariant fails."""


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
    if isinstance(value, bytes):
        return value.decode("ascii")
    if hasattr(value, "numpy"):
        return _safe(value.numpy())
    if hasattr(value, "tolist"):
        return _safe(value.tolist())
    if hasattr(value, "item"):
        return _safe(value.item())
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    absolute = _abs(path)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if absolute.exists():
        raise AISRepairError(f"refusing to overwrite artifact: {path}")
    encoded = (json.dumps(_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("ascii")
    temporary = absolute.with_suffix(absolute.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(absolute)


def _write_tensor(path: Path, value: Any, tf: Any) -> Mapping[str, Any]:
    absolute = _abs(path)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if absolute.exists():
        raise AISRepairError(f"refusing to overwrite artifact: {path}")
    tensor = tf.convert_to_tensor(value)
    encoded = bytes(tf.io.serialize_tensor(tensor).numpy())
    absolute.write_bytes(encoded)
    return {"path": path.as_posix(), "sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded), "dtype": tensor.dtype.name, "shape": list(tensor.shape)}


def _geometry_payload() -> Mapping[str, Any]:
    if _sha(GEOMETRY) != GEOMETRY_SHA256:
        raise AISRepairError("geometry artifact identity mismatch")
    payload = json.loads(_abs(GEOMETRY).read_text(encoding="utf-8"))
    if payload.get("status") != "GEOMETRY_DIAGNOSTIC_COMPLETED":
        raise AISRepairError("geometry artifact is incomplete")
    return payload


def _proposal_payload() -> Mapping[str, Any]:
    geometry = _geometry_payload()
    labels = ("plus", "minus")
    means = [geometry["representatives"][label]["position"] for label in labels]
    covariances = []
    # Host-side payload construction only.  Numerical inversion is performed in
    # TensorFlow inside each worker so this runner remains out of NumPy runtime.
    for label in labels:
        covariances.append(geometry["source_curvature"][label]["records"][-1]["precision"])
    return {"means": means, "precisions": covariances}


def _worker_environment() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS"):
        os.environ[name] = str(CORES_PER_WORKER)
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"


def _worker_run(task: Mapping[str, Any]) -> Mapping[str, Any]:
    cpu_ids = tuple(int(value) for value in task["cpu_ids"])
    os.sched_setaffinity(0, set(cpu_ids))
    _worker_environment()
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(CORES_PER_WORKER)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    if tf.config.list_physical_devices("GPU"):
        raise AISRepairError("AIS CPU worker found a visible GPU")
    from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import batch_native_complexity_posterior_target
    from bayesfilter.testing.annealed_importance_tf import run_linear_ais_fixed_hmc
    from bayesfilter.testing.importance_sampling_tf import gaussian_mixture_log_prob, sample_gaussian_mixture

    target = batch_native_complexity_posterior_target(20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh")
    if target.target_signature() != TARGET_SIGNATURE or target.adapter_signature() != ADAPTER_SIGNATURE:
        raise AISRepairError("AIS worker target identity mismatch")
    proposal = task["proposal"]
    means = tf.constant(proposal["means"], tf.float64)
    precisions = tf.constant(proposal["precisions"], tf.float64)
    covariances = tf.linalg.inv(precisions)
    probabilities = tf.constant((0.5, 0.5), tf.float64)
    chart_center = tf.reduce_mean(means, axis=0)
    displacement = means - chart_center
    pooled_covariance = tf.reduce_mean(covariances, axis=0) + tf.einsum("ni,nj->ij", displacement, displacement) / 2.0
    eigenvalues, eigenvectors = tf.linalg.eigh(pooled_covariance)
    factor = tf.matmul(eigenvectors * tf.sqrt(eigenvalues)[tf.newaxis, :], eigenvectors, transpose_b=True)
    chart_log_abs_determinant = tf.reduce_sum(tf.math.log(eigenvalues)) / 2.0
    theta_initial, component_labels = sample_gaussian_mixture(
        PATHS_PER_WORKER,
        probabilities,
        means,
        covariances,
        seed=tuple(int(value) for value in task["seed"]),
    )
    z_initial = tf.transpose(tf.linalg.solve(factor, tf.transpose(theta_initial - chart_center)))

    def proposal_log_prob(z: Any) -> Any:
        theta = chart_center + tf.matmul(z, factor, transpose_b=True)
        return gaussian_mixture_log_prob(theta, probabilities, means, covariances) + chart_log_abs_determinant

    def target_log_prob(z: Any) -> Any:
        theta = chart_center + tf.matmul(z, factor, transpose_b=True)
        value, _score, status = target.neutra_batch_log_prob_and_grad_status(theta)
        valid = tf.logical_and(tf.convert_to_tensor(status["status_code"], tf.int32) == 0, tf.convert_to_tensor(status["valid_pre_regularized_score"], tf.bool))
        invalid = tf.fill(tf.shape(value), tf.constant(float("nan"), tf.float64))
        return tf.where(valid, tf.convert_to_tensor(value, tf.float64) + chart_log_abs_determinant, invalid)

    started = time.perf_counter()
    result = run_linear_ais_fixed_hmc(
        proposal_log_prob,
        target_log_prob,
        z_initial,
        num_steps=int(task["num_steps"]),
        step_size=HMC_STEP_SIZE,
        num_leapfrog_steps=HMC_LEAPFROG,
        seed=tuple(int(value) for value in task["ais_seed"]),
        rejuvenation_interval=int(task["rejuvenation_interval"]),
        jit_compile=True,
    )
    terminal_theta = chart_center + tf.matmul(result["terminal_state"], factor, transpose_b=True)
    _terminal_value, _terminal_score, terminal_status = target.neutra_batch_log_prob_and_grad_status(terminal_theta)
    terminal_valid = tf.logical_and(tf.convert_to_tensor(terminal_status["status_code"], tf.int32) == 0, tf.convert_to_tensor(terminal_status["valid_pre_regularized_score"], tf.bool))
    all_finite = tf.logical_and(result["path_all_finite"], terminal_valid)
    initial_sign = theta_initial[:, 2] < 0.0
    terminal_sign = terminal_theta[:, 2] < 0.0
    return {
        "worker_index": int(task["worker_index"]),
        "cpu_ids": cpu_ids,
        "pid": os.getpid(),
        "target_signature": target.target_signature(),
        "adapter_signature": target.adapter_signature(),
        "num_steps": int(task["num_steps"]),
        "rejuvenation_interval": int(task["rejuvenation_interval"]),
        "rejuvenation_count": int(result["rejuvenation_count"].numpy()),
        "seed": tuple(task["seed"]),
        "ais_seed": tuple(task["ais_seed"]),
        "runtime_seconds": time.perf_counter() - started,
        "initial_theta": bytes(tf.io.serialize_tensor(theta_initial).numpy()),
        "component_labels": bytes(tf.io.serialize_tensor(component_labels).numpy()),
        "terminal_theta": bytes(tf.io.serialize_tensor(terminal_theta).numpy()),
        "initial_sign": bytes(tf.io.serialize_tensor(initial_sign).numpy()),
        "terminal_sign": bytes(tf.io.serialize_tensor(terminal_sign).numpy()),
        "log_weights": bytes(tf.io.serialize_tensor(result["log_weights"]).numpy()),
        "acceptance_fraction": bytes(tf.io.serialize_tensor(result["acceptance_fraction"]).numpy()),
        "all_finite": bytes(tf.io.serialize_tensor(all_finite).numpy()),
        "terminal_status_code": bytes(tf.io.serialize_tensor(terminal_status["status_code"]).numpy()),
        "terminal_valid_pre_regularized_score": bytes(tf.io.serialize_tensor(terminal_status["valid_pre_regularized_score"]).numpy()),
        "maximum_absolute_log_accept_ratio": bytes(tf.io.serialize_tensor(result["maximum_absolute_log_accept_ratio"]).numpy()),
    }


def _decode(tf: Any, payload: bytes, dtype: Any) -> Any:
    return tf.io.parse_tensor(payload, out_type=dtype)


def _manifest(
    mode: str,
    started: float,
    output_root: Path,
    tf: Any,
    tfp: Any,
) -> Mapping[str, Any]:
    return {
        "git_commit": subprocess.run(("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "git_dirty": bool(subprocess.run(("git", "status", "--porcelain"), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()),
        "command": " ".join(sys.argv),
        "mode": mode,
        "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "tensorflow_probability": tfp.__version__,
        "cpu_gpu_status": "CPU_ONLY_GPU_HIDDEN",
        "jit_compile": True,
        "wall_time_seconds": time.perf_counter() - started,
        "artifact_root": output_root.as_posix(),
        "plan_file": PLAN.as_posix(),
        "result_file": RESULT.as_posix(),
        "source_sha256": {"plan": _sha(PLAN), "runner": _sha(RUNNER), "ais_helper": _sha(AIS_HELPER), "importance_helper": _sha(IMPORTANCE_HELPER), "geometry": _sha(GEOMETRY), "direct_weight": _sha(DIRECT_WEIGHT)},
    }


def run_canary(
    *,
    output_root: Path = OUTPUT_ROOT,
    rejuvenation_interval: int = FULL_REJUVENATION_INTERVAL,
) -> Mapping[str, Any]:
    started = time.perf_counter()
    _worker_environment()
    if _sha(DIRECT_WEIGHT) != DIRECT_WEIGHT_SHA256:
        raise AISRepairError("direct-importance comparator identity mismatch")
    proposal = _proposal_payload()
    context = multiprocessing.get_context("spawn")
    task = {"worker_index": 0, "cpu_ids": (0, 1, 2, 3), "proposal": proposal, "num_steps": CANARY_STEPS, "rejuvenation_interval": rejuvenation_interval, "seed": (20260810, 7001), "ais_seed": (20260810, 7002)}
    with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
        row = executor.submit(_worker_run, task).result(timeout=CANARY_CAP_SECONDS)
    # Import only after the spawned worker exits, keeping parent framework state
    # out of process creation and making CPU/GPU isolation unambiguous.
    import tensorflow as tf
    import tensorflow_probability as tfp

    initial_theta = _decode(tf, row["initial_theta"], tf.float64)
    component_labels = _decode(tf, row["component_labels"], tf.int32)
    terminal_theta = _decode(tf, row["terminal_theta"], tf.float64)
    initial_sign = _decode(tf, row["initial_sign"], tf.bool)
    terminal_sign = _decode(tf, row["terminal_sign"], tf.bool)
    log_weights = _decode(tf, row["log_weights"], tf.float64)
    acceptance = _decode(tf, row["acceptance_fraction"], tf.float64)
    all_finite = _decode(tf, row["all_finite"], tf.bool)
    terminal_status_code = _decode(tf, row["terminal_status_code"], tf.int32)
    terminal_valid = _decode(tf, row["terminal_valid_pre_regularized_score"], tf.bool)
    max_log_accept = _decode(tf, row["maximum_absolute_log_accept_ratio"], tf.float64)
    passed = bool(tf.reduce_all(all_finite).numpy() and tf.reduce_mean(acceptance).numpy() >= 0.5)
    payload = {
        "schema": "bayesfilter.ssl_lstm.q20_physical_ais_repair.canary.v2",
        "status": "AIS_CANARY_PASSED" if passed else "AIS_CANARY_FAILED",
        "role": "bridge_correct_ais_mechanics_timing_canary",
        "configuration": {"paths": PATHS_PER_WORKER, "num_steps": CANARY_STEPS, "rejuvenation_interval": row["rejuvenation_interval"], "rejuvenation_count": row["rejuvenation_count"], "step_size": HMC_STEP_SIZE, "num_leapfrog_steps": HMC_LEAPFROG, "cpu_ids": row["cpu_ids"], "proposal_seed": row["seed"], "ais_seed": row["ais_seed"]},
        "worker": {"pid": row["pid"], "runtime_seconds": row["runtime_seconds"], "target_signature": row["target_signature"], "adapter_signature": row["adapter_signature"]},
        "all_paths_finite": tf.reduce_all(all_finite),
        "terminal_target_status_invalid_count": tf.reduce_sum(tf.cast(terminal_status_code != 0, tf.int32)),
        "terminal_target_valid_count": tf.reduce_sum(tf.cast(terminal_valid, tf.int32)),
        "acceptance_fraction_by_path": acceptance,
        "mean_acceptance_fraction": tf.reduce_mean(acceptance),
        "maximum_absolute_log_accept_ratio_by_path": max_log_accept,
        "initial_to_terminal_sign_changes": tf.reduce_sum(tf.cast(initial_sign != terminal_sign, tf.int32)),
        "receipts": {"initial_theta": _write_tensor(output_root / "ais-canary-initial-theta.tftensor", initial_theta, tf), "component_labels": _write_tensor(output_root / "ais-canary-component-labels.tftensor", component_labels, tf), "terminal_theta": _write_tensor(output_root / "ais-canary-terminal-theta.tftensor", terminal_theta, tf), "initial_sign": _write_tensor(output_root / "ais-canary-initial-sign.tftensor", initial_sign, tf), "terminal_sign": _write_tensor(output_root / "ais-canary-terminal-sign.tftensor", terminal_sign, tf), "log_weights": _write_tensor(output_root / "ais-canary-log-weights.tftensor", log_weights, tf), "acceptance_fraction": _write_tensor(output_root / "ais-canary-acceptance-fraction.tftensor", acceptance, tf), "all_finite": _write_tensor(output_root / "ais-canary-all-finite.tftensor", all_finite, tf), "terminal_status_code": _write_tensor(output_root / "ais-canary-terminal-status-code.tftensor", terminal_status_code, tf), "terminal_valid_pre_regularized_score": _write_tensor(output_root / "ais-canary-terminal-valid-pre-regularized-score.tftensor", terminal_valid, tf)},
        "run_manifest": _manifest("canary", started, output_root, tf, tfp),
        "nonclaims": ("four paths cannot estimate posterior mass", "canary acceptance and runtime are explanatory only", "two-mode proposal does not prove mode completeness"),
    }
    if time.perf_counter() - started > CANARY_CAP_SECONDS:
        raise AISRepairError("AIS canary wall-time cap breached")
    _write_json(output_root / "canary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("canary", "canary-sparse"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "canary-sparse":
        payload = run_canary(
            output_root=SPARSE_OUTPUT_ROOT,
            rejuvenation_interval=SPARSE_REJUVENATION_INTERVAL,
        )
    else:
        payload = run_canary()
    print(json.dumps({"mode": args.mode, "status": payload["status"]}, sort_keys=True))
