#!/usr/bin/env python3
"""Bounded physical-coordinate global repair diagnostics for SSL-LSTM q=20."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "docs" / "benchmarks"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

PLAN = Path("docs/plans/bayesfilter-ssl-lstm-q20-physical-global-repair-plan-2026-08-10.md")
RESULT = Path("docs/plans/bayesfilter-ssl-lstm-q20-physical-global-repair-result-2026-08-10.md")
RUNNER = Path("docs/benchmarks/run_ssl_lstm_q20_physical_global_repair_2026_08_10.py")
HELPER = Path("bayesfilter/testing/importance_sampling_tf.py")
REPLICA_HELPER = Path("bayesfilter/testing/replica_exchange_tf.py")
GEOMETRY = Path("docs/plans/artifacts/ssl-lstm-q20-seed-b-neutra-mode-failure-root-cause-2026-08-10/r1/geometry.json")
OUTPUT_ROOT = Path("docs/plans/artifacts/ssl-lstm-q20-physical-global-repair-2026-08-10/r1")

THREADS = 4
PARAMETER_DIM = 4
WORKERS = 25
ROWS_PER_WORKER = 4
ROWS_PER_BATCH = WORKERS * ROWS_PER_WORKER
CENTRAL_BATCHES = 8
SENSITIVITY_BATCHES = 2
WEIGHT_SAMPLE_CAP = 1200
WEIGHT_CAP_SECONDS = 1800.0
PHYSICAL_BETAS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
PHYSICAL_STEPS = tuple(0.05 / math.sqrt(beta) for beta in PHYSICAL_BETAS)
PHYSICAL_LEAPFROG = 8
PHYSICAL_TRANSITIONS = 12
PHYSICAL_CHAINS = 2
PHYSICAL_LOCAL_TRANSITIONS = 4
PHYSICAL_LOCAL_CAP_SECONDS = 2400.0
TRANSITION_CAP_SECONDS = 6000.0
TARGET_SIGNATURE = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
ADAPTER_SIGNATURE = "a8be6c212eb2a74ef926ccb9279871805949cae6e00c312a12525141638166f3"
GEOMETRY_SHA256: str | None = None


class PhysicalRepairError(RuntimeError):
    """Raised when a stage-2 evidence invariant fails."""


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
        raise PhysicalRepairError(f"refusing to overwrite artifact: {path}")
    encoded = (json.dumps(_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("ascii")
    temporary = absolute.with_suffix(absolute.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(absolute)


def _write_tensor(path: Path, value: Any, tf: Any) -> Mapping[str, Any]:
    absolute = _abs(path)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if absolute.exists():
        raise PhysicalRepairError(f"refusing to overwrite artifact: {path}")
    tensor = tf.convert_to_tensor(value)
    encoded = bytes(tf.io.serialize_tensor(tensor).numpy())
    absolute.write_bytes(encoded)
    return {"path": path.as_posix(), "sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded), "dtype": tensor.dtype.name, "shape": list(tensor.shape)}


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, _abs(path))
    if spec is None or spec.loader is None:
        raise PhysicalRepairError(f"cannot load module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configure() -> tuple[Any, Any]:
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.set_visible_devices([], "GPU")
    if tf.config.list_physical_devices("GPU"):
        raise PhysicalRepairError("physical repair is CPU-only but a GPU is visible")
    from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import batch_native_complexity_posterior_target
    target = batch_native_complexity_posterior_target(20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh")
    if target.target_signature() != TARGET_SIGNATURE or target.adapter_signature() != ADAPTER_SIGNATURE:
        raise PhysicalRepairError("live target signature mismatch")
    return tf, target


def _manifest(mode: str, started: float, tf: Any, tfp: Any) -> Mapping[str, Any]:
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
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_threads": THREADS,
        "jit_compile": True,
        "wall_time_seconds": time.perf_counter() - started,
        "artifact_root": OUTPUT_ROOT.as_posix(),
        "plan_file": PLAN.as_posix(),
        "result_file": RESULT.as_posix(),
        "source_sha256": {"plan": _sha(PLAN), "runner": _sha(RUNNER), "importance_helper": _sha(HELPER), "replica_helper": _sha(REPLICA_HELPER), "geometry": _sha(GEOMETRY)},
    }


def _load_geometry() -> Mapping[str, Any]:
    global GEOMETRY_SHA256
    payload = json.loads(_abs(GEOMETRY).read_text(encoding="utf-8"))
    if payload.get("status") != "GEOMETRY_DIAGNOSTIC_COMPLETED":
        raise PhysicalRepairError("geometry artifact is not complete")
    if not payload.get("source_curvature", {}).get("plus", {}).get("spd") or not payload.get("source_curvature", {}).get("minus", {}).get("spd"):
        raise PhysicalRepairError("both source local precisions must be SPD")
    GEOMETRY_SHA256 = _sha(GEOMETRY)
    return payload


def _target_parity(tf: Any, target: Any, geometry: Mapping[str, Any]) -> Mapping[str, Any]:
    labels = ("plus", "minus")
    rows = tf.constant(
        [geometry["representatives"][label]["position"] for label in labels],
        tf.float64,
    )
    values, scores, status = target.neutra_batch_log_prob_and_grad_status(rows)
    expected_values = tf.constant(
        [geometry["representatives"][label]["log_prob"] for label in labels],
        tf.float64,
    )
    expected_score_norms = tf.constant(
        [geometry["representatives"][label]["score_inf_norm"] for label in labels],
        tf.float64,
    )
    value_residual = tf.reduce_max(tf.abs(values - expected_values))
    score_norms = tf.reduce_max(tf.abs(scores), axis=1)
    score_norm_residual = tf.reduce_max(tf.abs(score_norms - expected_score_norms))
    valid = tf.logical_and(
        tf.convert_to_tensor(status["status_code"], tf.int32) == 0,
        tf.convert_to_tensor(status["valid_pre_regularized_score"], tf.bool),
    )
    passed = bool(
        tf.reduce_all(valid).numpy()
        and value_residual <= tf.constant(5.0e-7, tf.float64)
        and score_norm_residual <= tf.constant(5.0e-7, tf.float64)
    )
    result = {
        "status": "PASSED" if passed else "FAILED",
        "values": values,
        "expected_values": expected_values,
        "maximum_absolute_value_residual": value_residual,
        "score_inf_norms": score_norms,
        "expected_score_inf_norms": expected_score_norms,
        "maximum_absolute_score_norm_residual": score_norm_residual,
        "tolerance": 5.0e-7,
        "all_status_valid": tf.reduce_all(valid),
    }
    if not passed:
        raise PhysicalRepairError("current exact target failed source representative parity")
    return result


def _physical_chart(tf: Any, geometry: Mapping[str, Any]) -> Mapping[str, Any]:
    labels = ("plus", "minus")
    centers = tf.constant([geometry["representatives"][label]["position"] for label in labels], tf.float64)
    precisions = tf.stack([tf.constant(geometry["source_curvature"][label]["records"][-1]["precision"], tf.float64) for label in labels])
    covariances = tf.linalg.inv(precisions)
    center = tf.reduce_mean(centers, axis=0)
    displacement = centers - center
    pooled_covariance = tf.reduce_mean(covariances, axis=0) + tf.einsum("ni,nj->ij", displacement, displacement) / 2.0
    eigenvalues, eigenvectors = tf.linalg.eigh(pooled_covariance)
    factor = tf.matmul(eigenvectors * tf.sqrt(eigenvalues)[tf.newaxis, :], eigenvectors, transpose_b=True)
    latent_centers = tf.transpose(tf.linalg.solve(factor, tf.transpose(centers - center)))
    mapped_precisions = tf.stack([tf.matmul(factor, tf.matmul(precision, factor), transpose_a=True) for precision in precisions])
    mapped_eigenvalues = tf.linalg.eigvalsh(mapped_precisions)
    return {
        "center": center,
        "factor": factor,
        "source_centers": centers,
        "source_covariances": covariances,
        "pooled_covariance": pooled_covariance,
        "latent_centers": latent_centers,
        "mapped_precision_eigenvalues": mapped_eigenvalues,
        "mapped_mode_distance": tf.linalg.norm(latent_centers[0] - latent_centers[1]),
        "provenance": "equal_weight_law_total_covariance_of_two_checked_local_gaussians; warm_start_only",
    }


def _make_target_fn(tf: Any, target: Any) -> Any:
    def target_fn(state: Any) -> Any:
        rows = tf.reshape(tf.convert_to_tensor(state, tf.float64), (-1, PARAMETER_DIM))
        values, _scores, _status = target.neutra_batch_log_prob_and_grad_status(rows)
        return tf.reshape(tf.convert_to_tensor(values, tf.float64), tf.shape(state)[:-1])

    return target_fn


def _pool_config() -> Any:
    from bayesfilter.inference.tf_batch_value_score_pool import TFBatchValueScorePoolConfig

    return TFBatchValueScorePoolConfig(
        factory_path=(
            "bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf:"
            "batch_native_complexity_target_worker_factory"
        ),
        factory_config={
            "q": 20,
            "principal_sqrt_backend": "tensorflow_eigh",
            "jit_compile": True,
        },
        dimension=PARAMETER_DIM,
        worker_count=WORKERS,
        cores_per_worker=1,
        batch_sizes=(ROWS_PER_WORKER,),
        batch_per_worker=ROWS_PER_WORKER,
        worker_cpu_ids=tuple(range(64, 64 + WORKERS)),
        timeout_seconds=900.0,
    )


def _weight_batch(
    tf: Any,
    helper: Any,
    pool: Any,
    *,
    means: Any,
    covariances: Any,
    covariance_scale: float,
    seed: tuple[int, int],
    label: str,
) -> Mapping[str, Any]:
    probabilities = tf.constant((0.5, 0.5), tf.float64)
    scaled_covariances = tf.convert_to_tensor(covariances, tf.float64) * tf.constant(
        covariance_scale, tf.float64
    )
    rows, component_labels = helper.sample_gaussian_mixture(
        ROWS_PER_BATCH,
        probabilities,
        means,
        scaled_covariances,
        seed=seed,
    )
    proposal_log_prob = helper.gaussian_mixture_log_prob(
        rows, probabilities, means, scaled_covariances
    )
    target_values, target_scores, status, metadata = pool.evaluate_with_status(
        rows, request_id=label
    )
    valid = tf.logical_and(
        tf.convert_to_tensor(status["status_code"], tf.int32) == 0,
        tf.convert_to_tensor(status["valid_pre_regularized_score"], tf.bool),
    )
    invalid_count = int(tf.reduce_sum(tf.cast(tf.logical_not(valid), tf.int32)).numpy())
    if invalid_count:
        return {
            "status": "TARGET_INVALID",
            "label": label,
            "seed": seed,
            "covariance_scale": covariance_scale,
            "invalid_count": invalid_count,
            "status_counts": {
                "nonzero_status_code": tf.reduce_sum(
                    tf.cast(tf.convert_to_tensor(status["status_code"], tf.int32) != 0, tf.int32)
                ),
                "invalid_pre_regularized_score": tf.reduce_sum(
                    tf.cast(tf.logical_not(tf.convert_to_tensor(status["valid_pre_regularized_score"], tf.bool)), tf.int32)
                ),
            },
            "rows": _write_tensor(OUTPUT_ROOT / f"{label}-rows.tftensor", rows, tf),
            "component_labels": _write_tensor(
                OUTPUT_ROOT / f"{label}-component-labels.tftensor", component_labels, tf
            ),
            "target_status_code": _write_tensor(
                OUTPUT_ROOT / f"{label}-target-status-code.tftensor",
                status["status_code"],
                tf,
            ),
            "metadata": metadata,
        }
    diagnostics = helper.self_normalized_importance_diagnostics(
        target_values, proposal_log_prob, rows[:, 2] < 0.0
    )
    receipts = {
        "rows": _write_tensor(OUTPUT_ROOT / f"{label}-rows.tftensor", rows, tf),
        "component_labels": _write_tensor(
            OUTPUT_ROOT / f"{label}-component-labels.tftensor", component_labels, tf
        ),
        "target_values": _write_tensor(
            OUTPUT_ROOT / f"{label}-target-values.tftensor", target_values, tf
        ),
        "target_scores": _write_tensor(
            OUTPUT_ROOT / f"{label}-target-scores.tftensor", target_scores, tf
        ),
        "proposal_log_prob": _write_tensor(
            OUTPUT_ROOT / f"{label}-proposal-log-prob.tftensor", proposal_log_prob, tf
        ),
        "log_weights": _write_tensor(
            OUTPUT_ROOT / f"{label}-log-weights.tftensor", diagnostics["log_weights"], tf
        ),
        "normalized_weights": _write_tensor(
            OUTPUT_ROOT / f"{label}-normalized-weights.tftensor",
            diagnostics["normalized_weights"],
            tf,
        ),
    }
    return {
        "status": "COMPLETED",
        "label": label,
        "seed": seed,
        "covariance_scale": covariance_scale,
        "invalid_count": 0,
        "component_zero_count": tf.reduce_sum(tf.cast(component_labels == 0, tf.int32)),
        "diagnostics": {
            name: value
            for name, value in diagnostics.items()
            if name not in {"log_weights", "normalized_weights"}
        },
        "receipts": receipts,
        "metadata": metadata,
    }


def run_weights(*, canary_only: bool) -> Mapping[str, Any]:
    started = time.perf_counter()
    tf, target = _configure()
    import tensorflow_probability as tfp
    from bayesfilter.inference.tf_batch_value_score_pool import TFBatchValueScorePool

    helper = _load_module(HELPER, "physical_global_importance_sampling_helper")
    geometry = _load_geometry()
    parity = _target_parity(tf, target, geometry)
    chart = _physical_chart(tf, geometry)
    if canary_only:
        specifications = (("weight-canary", 1.0, (20260810, 5001)),)
    else:
        canary_path = OUTPUT_ROOT / "weight-canary.json"
        if not _abs(canary_path).exists():
            raise PhysicalRepairError("passed weight-canary artifact is required")
        canary = json.loads(_abs(canary_path).read_text(encoding="utf-8"))
        if canary.get("status") != "WEIGHT_CANARY_PASSED":
            raise PhysicalRepairError("weight canary did not pass")
        specifications = tuple(
            (f"weight-central-{index:02d}", 1.0, (20260810, 5100 + index))
            for index in range(CENTRAL_BATCHES)
        ) + tuple(
            (f"weight-scale-{scale:g}-{index:02d}", scale, (20260810, seed_base + index))
            for scale, seed_base in ((0.5, 5200), (2.0, 5300))
            for index in range(SENSITIVITY_BATCHES)
        )
    if len(specifications) * ROWS_PER_BATCH > WEIGHT_SAMPLE_CAP:
        raise PhysicalRepairError("weight row cap would be breached")
    batches = []
    with TFBatchValueScorePool(_pool_config()) as pool:
        for label, scale, seed in specifications:
            batches.append(
                _weight_batch(
                    tf,
                    helper,
                    pool,
                    means=chart["source_centers"],
                    covariances=chart["source_covariances"],
                    covariance_scale=scale,
                    seed=seed,
                    label=label,
                )
            )
            progress = {
                "schema": "bayesfilter.ssl_lstm.q20_physical_global_repair.weight_progress.v1",
                "status": "RUNNING",
                "completed_batch_count": len(batches),
                "planned_batch_count": len(specifications),
                "batches": batches,
            }
            progress_path = OUTPUT_ROOT / (
                "weight-canary-progress.json" if canary_only else "weights-progress.json"
            )
            absolute_progress = _abs(progress_path)
            absolute_progress.parent.mkdir(parents=True, exist_ok=True)
            temporary = absolute_progress.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(_safe(progress), sort_keys=True, indent=2, allow_nan=False) + "\n",
                encoding="ascii",
            )
            temporary.replace(absolute_progress)
    all_completed = all(row["status"] == "COMPLETED" for row in batches)
    if canary_only:
        status = "WEIGHT_CANARY_PASSED" if all_completed else "WEIGHT_CANARY_FAILED"
        evidence = {"all_target_rows_valid": all_completed}
    else:
        status, evidence = _aggregate_weight_evidence(tf, helper, batches)
    payload = {
        "schema": "bayesfilter.ssl_lstm.q20_physical_global_repair.weights.v1",
        "status": status,
        "role": "corrected_two_known_mode_importance_mass_diagnostic",
        "canary_only": canary_only,
        "configuration": {
            "proposal_component_probabilities": (0.5, 0.5),
            "proposal_means": chart["source_centers"],
            "proposal_covariances": chart["source_covariances"],
            "rows_per_batch": ROWS_PER_BATCH,
            "batch_count": len(specifications),
            "worker_topology": "25 pinned CPU workers x 4 batch-native XLA rows",
        },
        "target_parity": parity,
        "batches": batches,
        "evidence": evidence,
        "run_manifest": _manifest("weight-canary" if canary_only else "weights", started, tf, tfp),
        "nonclaims": (
            "proposal covers only two known regions and cannot prove mode completeness",
            "local Gaussian proposal is corrected by exact target/proposal weights",
            "no HMC transition or NeuTra repair claim",
            "failed weight gates trigger AIS or SMC",
        ),
    }
    if time.perf_counter() - started > WEIGHT_CAP_SECONDS:
        raise PhysicalRepairError("weight wall-time cap breached")
    output = OUTPUT_ROOT / ("weight-canary.json" if canary_only else "weights.json")
    _write_json(output, payload)
    return payload


def _aggregate_weight_evidence(
    tf: Any, helper: Any, batches: list[Mapping[str, Any]]
) -> tuple[str, Mapping[str, Any]]:
    all_completed = all(row.get("status") == "COMPLETED" for row in batches)
    central = [row for row in batches if float(row["covariance_scale"]) == 1.0]
    narrow = [row for row in batches if float(row["covariance_scale"]) == 0.5]
    broad = [row for row in batches if float(row["covariance_scale"]) == 2.0]
    if (len(central), len(narrow), len(broad)) != (
        CENTRAL_BATCHES,
        SENSITIVITY_BATCHES,
        SENSITIVITY_BATCHES,
    ):
        raise PhysicalRepairError("weight batch scale/count ledger mismatch")
    if not all_completed:
        return "WEIGHT_DIAGNOSTIC_FAILED", {
            "gates": {"all_target_rows_valid": False},
            "failure": "one or more target batches were incomplete or invalid",
        }

    def stack(rows: list[Mapping[str, Any]], name: str) -> Any:
        return tf.stack(
            [
                tf.convert_to_tensor(row["diagnostics"][name], tf.float64)
                for row in rows
            ]
        )

    central_estimates = stack(central, "negative_region_probability")
    interval = helper.independent_batch_interval(central_estimates)
    pooled_ess_fraction = tf.reduce_mean(
        stack(central, "effective_sample_size_fraction")
    )
    maximum_weight = tf.reduce_max(stack(central, "maximum_normalized_weight"))
    sensitivity_means = {
        "scale_0_5": tf.reduce_mean(
            stack(narrow, "negative_region_probability")
        ),
        "scale_2": tf.reduce_mean(
            stack(broad, "negative_region_probability")
        ),
    }
    sensitivity_max_difference = tf.reduce_max(
        tf.abs(tf.stack(tuple(sensitivity_means.values())) - interval["mean"])
    )
    gates = {
        "all_target_rows_valid": True,
        "mean_batch_ess_fraction_at_least_0_20": bool(
            pooled_ess_fraction.numpy() >= 0.20
        ),
        "maximum_batch_weight_at_most_0_05": bool(
            maximum_weight.numpy() <= 0.05
        ),
        "interval_half_width_at_most_0_10": bool(
            interval["half_width"].numpy() <= 0.10
        ),
        "scale_sensitivity_at_most_0_10": bool(
            sensitivity_max_difference.numpy() <= 0.10
        ),
    }
    status = (
        "WEIGHT_DIAGNOSTIC_PASSED"
        if all(gates.values())
        else "WEIGHT_DIAGNOSTIC_FAILED"
    )
    return status, {
        "central_batch_estimates": central_estimates,
        "independent_batch_interval": interval,
        "mean_central_batch_ess_fraction": pooled_ess_fraction,
        "maximum_central_batch_normalized_weight": maximum_weight,
        "sensitivity_means": sensitivity_means,
        "sensitivity_max_difference": sensitivity_max_difference,
        "gates": gates,
    }


def run_weight_aggregate() -> Mapping[str, Any]:
    """Recover terminal evidence from completed immutable batch receipts."""

    started = time.perf_counter()
    tf, target = _configure()
    import tensorflow_probability as tfp

    helper = _load_module(HELPER, "physical_global_importance_aggregate_helper")
    progress_path = OUTPUT_ROOT / "weights-progress.json"
    if not _abs(progress_path).exists() or _abs(OUTPUT_ROOT / "weights.json").exists():
        raise PhysicalRepairError("aggregate requires progress and no terminal artifact")
    progress = json.loads(_abs(progress_path).read_text(encoding="utf-8"))
    batches = progress.get("batches")
    if (
        not isinstance(batches, list)
        or len(batches) != CENTRAL_BATCHES + 2 * SENSITIVITY_BATCHES
        or int(progress.get("completed_batch_count", -1)) != len(batches)
    ):
        raise PhysicalRepairError("progress artifact is not a completed campaign")
    for batch in batches:
        if batch.get("status") != "COMPLETED":
            raise PhysicalRepairError("cannot aggregate an incomplete target batch")
        metadata = batch.get("metadata", {})
        startup_rows = metadata.get("startup_worker_metadata")
        if not isinstance(startup_rows, list) or len(startup_rows) != WORKERS:
            raise PhysicalRepairError("batch startup worker ledger is incomplete")
        if any(row.get("target_signature") != TARGET_SIGNATURE for row in startup_rows):
            raise PhysicalRepairError("batch target signature mismatch")
        if any(row.get("adapter_signature") != ADAPTER_SIGNATURE for row in startup_rows):
            raise PhysicalRepairError("batch adapter signature mismatch")
        if set(int(value) for value in metadata.get("worker_assigned_cpu_ids", ())) != set(
            range(64, 64 + WORKERS)
        ):
            raise PhysicalRepairError("batch worker CPU ledger mismatch")
        for receipt in batch.get("receipts", {}).values():
            path = Path(str(receipt["path"]))
            encoded = _abs(path).read_bytes()
            if (
                hashlib.sha256(encoded).hexdigest() != str(receipt["sha256"])
                or len(encoded) != int(receipt["bytes"])
            ):
                raise PhysicalRepairError(f"weight receipt mismatch: {path}")
    status, evidence = _aggregate_weight_evidence(tf, helper, batches)
    geometry = _load_geometry()
    parity = _target_parity(tf, target, geometry)
    chart = _physical_chart(tf, geometry)
    payload = {
        "schema": "bayesfilter.ssl_lstm.q20_physical_global_repair.weights.v1",
        "status": status,
        "role": "corrected_two_known_mode_importance_mass_diagnostic",
        "canary_only": False,
        "configuration": {
            "proposal_component_probabilities": (0.5, 0.5),
            "proposal_means": chart["source_centers"],
            "proposal_covariances": chart["source_covariances"],
            "rows_per_batch": ROWS_PER_BATCH,
            "batch_count": len(batches),
            "worker_topology": "25 pinned CPU workers x 4 batch-native XLA rows",
        },
        "target_parity": parity,
        "batches": batches,
        "evidence": evidence,
        "aggregation_recovery": {
            "source_progress": progress_path.as_posix(),
            "source_progress_sha256": _sha(progress_path),
            "reason": "terminal-only tf.constant(list_of_scalar_tensors) TypeError after all target batches completed",
            "target_rows_recomputed": 0,
            "all_batch_receipts_reverified": True,
        },
        "run_manifest": _manifest("weights-aggregate", started, tf, tfp),
        "nonclaims": (
            "proposal covers only two known regions and cannot prove mode completeness",
            "local Gaussian proposal is corrected by exact target/proposal weights",
            "no HMC transition or NeuTra repair claim",
            "failed weight gates trigger AIS or SMC",
        ),
    }
    _write_json(OUTPUT_ROOT / "weights.json", payload)
    return payload


def run_physical_transition() -> Mapping[str, Any]:
    local_path = OUTPUT_ROOT / "physical-local.json"
    if not _abs(local_path).exists():
        raise PhysicalRepairError("passed physical-local artifact is required")
    local = json.loads(_abs(local_path).read_text(encoding="utf-8"))
    if local.get("status") != "PHYSICAL_LOCAL_CANARY_PASSED":
        raise PhysicalRepairError("physical local canary did not pass")
    started = time.perf_counter()
    tf, target = _configure()
    import tensorflow_probability as tfp
    replica_helper = _load_module(REPLICA_HELPER, "physical_global_replica_exchange_helper")
    geometry = _load_geometry()
    parity = _target_parity(tf, target, geometry)
    chart = _physical_chart(tf, geometry)
    center = chart["center"]
    factor = chart["factor"]
    latent_centers = chart["latent_centers"]
    chart_log_abs_determinant = tf.reduce_sum(
        tf.math.log(tf.linalg.eigvalsh(factor))
    )
    initial_latent = tf.repeat(
        latent_centers[tf.newaxis, :, :], len(PHYSICAL_BETAS), axis=0
    )

    def latent_target(state: Any) -> Any:
        z = tf.reshape(tf.convert_to_tensor(state, tf.float64), (-1, PARAMETER_DIM))
        theta = center + tf.matmul(z, factor, transpose_b=True)
        value, _score, _status = target.neutra_batch_log_prob_and_grad_status(theta)
        return tf.reshape(
            tf.convert_to_tensor(value, tf.float64) + chart_log_abs_determinant,
            tf.shape(state)[:-1],
        )

    trace = replica_helper.run_replica_exchange_fixed_hmc(
        latent_target,
        initial_latent,
        inverse_temperatures=PHYSICAL_BETAS,
        step_sizes=PHYSICAL_STEPS,
        num_leapfrog_steps=PHYSICAL_LEAPFROG,
        num_steps=PHYSICAL_TRANSITIONS,
        seed=(20260810, 4101),
        jit_compile=True,
    )
    theta = center + tf.matmul(tf.reshape(trace["replica_states"], (-1, PARAMETER_DIM)), factor, transpose_b=True)
    theta = tf.reshape(theta, (PHYSICAL_TRANSITIONS, len(PHYSICAL_BETAS), PHYSICAL_CHAINS, PARAMETER_DIM))
    signs = theta[..., 2] < 0.0
    pre_theta = center + tf.matmul(tf.reshape(trace["pre_swap_replica_states"], (-1, PARAMETER_DIM)), factor, transpose_b=True)
    pre_theta = tf.reshape(pre_theta, tf.shape(theta))
    pre_signs = pre_theta[..., 2] < 0.0
    initial_signs = tf.broadcast_to(
        chart["source_centers"][:, 2] < 0.0,
        (len(PHYSICAL_BETAS), PHYSICAL_CHAINS),
    )
    previous_post_signs = tf.concat(
        (initial_signs[tf.newaxis, ...], signs[:-1]), axis=0
    )
    local_hmc_sign_changes = pre_signs != previous_post_signs
    accepted_states = tf.reshape(theta, (-1, PARAMETER_DIM))
    _values, _scores, statuses = target.neutra_batch_log_prob_and_grad_status(accepted_states)
    invalid = tf.logical_or(tf.convert_to_tensor(statuses["status_code"], tf.int32) != 0, tf.logical_not(tf.convert_to_tensor(statuses["valid_pre_regularized_score"], tf.bool)))
    proposed = tf.reduce_sum(tf.cast(trace["swap_is_proposed_adjacent"], tf.int32), axis=(0, 2))
    accepted = tf.reduce_sum(tf.cast(trace["swap_is_accepted_adjacent"], tf.int32), axis=(0, 2))
    payload = {
        "schema": "bayesfilter.ssl_lstm.q20_physical_global_repair.transition.v1",
        "status": "PHYSICAL_TRANSITION_CANARY_COMPLETED",
        "role": "physical_coordinate_replica_exchange_mechanics_and_global_transition_diagnostic",
        "configuration": {"inverse_temperatures": PHYSICAL_BETAS, "step_sizes": PHYSICAL_STEPS, "num_leapfrog_steps": PHYSICAL_LEAPFROG, "transitions": PHYSICAL_TRANSITIONS, "chains": PHYSICAL_CHAINS, "seed": (20260810, 4101), "initialization": "both checked physical stationary representatives"},
        "chart": {"center": center, "factor": factor, "source_centers": chart["source_centers"], "source_covariances": chart["source_covariances"], "pooled_covariance": chart["pooled_covariance"], "latent_centers": latent_centers, "mapped_precision_eigenvalues": chart["mapped_precision_eigenvalues"], "mapped_mode_distance": chart["mapped_mode_distance"], "provenance": chart["provenance"]},
        "finite": bool(replica_helper.replica_exchange_finite(trace).numpy()),
        "accepted_state_target_status_invalid_count": int(tf.reduce_sum(tf.cast(invalid, tf.int32)).numpy()),
        "hmc_acceptance_by_temperature": tf.reduce_mean(tf.cast(trace["hmc_is_accepted"], tf.float64), axis=(0, 2)),
        "adjacent_swap_proposals": proposed,
        "adjacent_swap_acceptances": accepted,
        "adjacent_swap_acceptance_rates": tf.math.divide_no_nan(tf.cast(accepted, tf.float64), tf.cast(proposed, tf.float64)),
        "post_swap_cold_sign_transitions": int(tf.reduce_sum(tf.cast(signs[1:, 0] != signs[:-1, 0], tf.int32)).numpy()),
        "local_hmc_hot_sign_changes": int(tf.reduce_sum(tf.cast(local_hmc_sign_changes[:, 1:], tf.int32)).numpy()),
        "local_hmc_cold_sign_changes": int(tf.reduce_sum(tf.cast(local_hmc_sign_changes[:, 0], tf.int32)).numpy()),
        "pre_swap_hot_sign_fraction": tf.reduce_mean(tf.cast(pre_signs[:, 1:], tf.float64)),
        "completed_round_trips": tf.reduce_sum(trace["round_trip_returns"]),
        "trace_receipts": {name: _write_tensor(OUTPUT_ROOT / f"physical-transition-{name}.tftensor", trace[name], tf) for name in ("replica_states", "pre_swap_replica_states", "hmc_is_accepted", "hmc_log_accept_ratio", "swap_is_proposed_adjacent", "swap_is_accepted_adjacent", "swap_is_accepted_matrix", "potential_energy", "replica_identities_at_temperature", "temperature_position_by_chain_identity", "round_trip_returns")},
        "physical_signs": _write_tensor(OUTPUT_ROOT / "physical-transition-signs.tftensor", signs, tf),
        "pre_swap_physical_signs": _write_tensor(OUTPUT_ROOT / "physical-transition-pre-swap-signs.tftensor", pre_signs, tf),
        "bindings": {"target_signature": target.target_signature(), "target_adapter_signature": target.adapter_signature(), "target_parity": parity, "geometry_sha256": GEOMETRY_SHA256},
        "local_gate": {"path": local_path.as_posix(), "sha256": _sha(local_path)},
        "run_manifest": _manifest("physical-transition", started, tf, tfp),
        "nonclaims": ("not a stationary posterior archive", "no mode-weight estimate", "two sign regions not exhaustive", "transition/acceptance/runtimes descriptive unless later uncertainty evidence passes"),
    }
    if time.perf_counter() - started > TRANSITION_CAP_SECONDS:
        raise PhysicalRepairError("physical transition wall-time cap breached")
    _write_json(OUTPUT_ROOT / "physical-transition.json", payload)
    return payload


def run_physical_local() -> Mapping[str, Any]:
    """Check the shared physical chart in both known regions before tempering."""

    started = time.perf_counter()
    tf, target = _configure()
    import tensorflow_probability as tfp

    geometry = _load_geometry()
    parity = _target_parity(tf, target, geometry)
    chart = _physical_chart(tf, geometry)
    center = chart["center"]
    factor = chart["factor"]
    initial = chart["latent_centers"]
    chart_log_abs_determinant = tf.reduce_sum(
        tf.math.log(tf.linalg.eigvalsh(factor))
    )

    def latent_target(state: Any) -> Any:
        z = tf.reshape(tf.convert_to_tensor(state, tf.float64), (-1, PARAMETER_DIM))
        theta = center + tf.matmul(z, factor, transpose_b=True)
        value, _score, _status = target.neutra_batch_log_prob_and_grad_status(theta)
        return tf.reshape(
            tf.convert_to_tensor(value, tf.float64) + chart_log_abs_determinant,
            tf.shape(state)[:-1],
        )

    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=latent_target,
        step_size=tf.constant(PHYSICAL_STEPS[0], tf.float64),
        num_leapfrog_steps=PHYSICAL_LEAPFROG,
    )

    def trace_fn(_state: Any, results: Any) -> Mapping[str, Any]:
        return {
            "is_accepted": results.is_accepted,
            "log_accept_ratio": results.log_accept_ratio,
            "accepted_target_log_prob": results.accepted_results.target_log_prob,
            "proposed_target_log_prob": results.proposed_results.target_log_prob,
        }

    @tf.function(jit_compile=True, reduce_retracing=False)
    def sample() -> Any:
        return tfp.mcmc.sample_chain(
            num_results=PHYSICAL_LOCAL_TRANSITIONS,
            num_burnin_steps=0,
            current_state=initial,
            kernel=kernel,
            trace_fn=trace_fn,
            seed=tf.constant((20260810, 4001), tf.int32),
        )

    samples, trace = sample()
    theta = center + tf.matmul(
        tf.reshape(samples, (-1, PARAMETER_DIM)), factor, transpose_b=True
    )
    theta = tf.reshape(theta, (PHYSICAL_LOCAL_TRANSITIONS, 2, PARAMETER_DIM))
    _values, _scores, status = target.neutra_batch_log_prob_and_grad_status(
        tf.reshape(theta, (-1, PARAMETER_DIM))
    )
    invalid = tf.logical_or(
        tf.convert_to_tensor(status["status_code"], tf.int32) != 0,
        tf.logical_not(
            tf.convert_to_tensor(status["valid_pre_regularized_score"], tf.bool)
        ),
    )
    finite = tf.reduce_all(
        tf.stack(
            (
                tf.reduce_all(tf.math.is_finite(samples)),
                tf.reduce_all(tf.math.is_finite(trace["log_accept_ratio"])),
                tf.reduce_all(tf.math.is_finite(trace["accepted_target_log_prob"])),
                tf.reduce_all(tf.math.is_finite(trace["proposed_target_log_prob"])),
            )
        )
    )
    acceptance = tf.reduce_mean(tf.cast(trace["is_accepted"], tf.float64), axis=0)
    passed = bool(
        finite.numpy()
        and tf.reduce_sum(tf.cast(invalid, tf.int32)).numpy() == 0
        and tf.reduce_all(acceptance >= 0.5).numpy()
    )
    payload = {
        "schema": "bayesfilter.ssl_lstm.q20_physical_global_repair.local.v1",
        "status": "PHYSICAL_LOCAL_CANARY_PASSED" if passed else "PHYSICAL_LOCAL_CANARY_FAILED",
        "role": "shared_physical_chart_two_region_local_hmc_gate",
        "configuration": {
            "step_size": PHYSICAL_STEPS[0],
            "num_leapfrog_steps": PHYSICAL_LEAPFROG,
            "transitions": PHYSICAL_LOCAL_TRANSITIONS,
            "seed": (20260810, 4001),
        },
        "chart": {
            "center": center,
            "factor": factor,
            "latent_centers": chart["latent_centers"],
            "mapped_precision_eigenvalues": chart["mapped_precision_eigenvalues"],
            "mapped_mode_distance": chart["mapped_mode_distance"],
            "provenance": chart["provenance"],
        },
        "finite": finite,
        "accepted_state_target_status_invalid_count": tf.reduce_sum(
            tf.cast(invalid, tf.int32)
        ),
        "acceptance_by_initial_region_plus_minus": acceptance,
        "maximum_absolute_log_accept_ratio": tf.reduce_max(
            tf.abs(trace["log_accept_ratio"])
        ),
        "receipts": {
            "samples": _write_tensor(
                OUTPUT_ROOT / "physical-local-samples.tftensor", samples, tf
            ),
            "is_accepted": _write_tensor(
                OUTPUT_ROOT / "physical-local-is-accepted.tftensor",
                trace["is_accepted"],
                tf,
            ),
            "log_accept_ratio": _write_tensor(
                OUTPUT_ROOT / "physical-local-log-accept-ratio.tftensor",
                trace["log_accept_ratio"],
                tf,
            ),
            "proposed_target_log_prob": _write_tensor(
                OUTPUT_ROOT / "physical-local-proposed-target-log-prob.tftensor",
                trace["proposed_target_log_prob"],
                tf,
            ),
        },
        "bindings": {
            "target_signature": target.target_signature(),
            "target_adapter_signature": target.adapter_signature(),
            "target_parity": parity,
            "geometry_sha256": GEOMETRY_SHA256,
        },
        "run_manifest": _manifest("physical-local", started, tf, tfp),
        "nonclaims": (
            "four local transitions are not warm-up or posterior sampling",
            "passing local integration does not establish global transitions",
        ),
    }
    if time.perf_counter() - started > PHYSICAL_LOCAL_CAP_SECONDS:
        raise PhysicalRepairError("physical local wall-time cap breached")
    _write_json(OUTPUT_ROOT / "physical-local.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "weight-canary",
            "weights",
            "weights-aggregate",
            "physical-local",
            "physical-transition",
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = (
        run_weights(canary_only=True)
        if args.mode == "weight-canary"
        else run_weights(canary_only=False)
        if args.mode == "weights"
        else run_weight_aggregate()
        if args.mode == "weights-aggregate"
        else run_physical_local()
        if args.mode == "physical-local"
        else run_physical_transition()
    )
    print(json.dumps({"mode": args.mode, "status": result["status"]}, sort_keys=True))
