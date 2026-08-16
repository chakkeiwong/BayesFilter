#!/usr/bin/env python3
"""Calibrated q=20 posterior-mean versus true output-law comparison."""

from __future__ import annotations

import argparse
import hashlib
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
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BENCHMARKS = ROOT / "docs" / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

PLAN = Path(
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-seed-b-predictive-equivalence-repair-plan-2026-08-08.md"
)
RUNNER = Path(
    "docs/benchmarks/"
    "run_ssl_lstm_q20_seed_b_predictive_equivalence_repair_2026_08_08.py"
)
DEFAULT_ROOT = Path(
    "docs/plans/artifacts/"
    "ssl-lstm-q20-seed-b-predictive-equivalence-repair-2026-08-08/r3"
)
ARCHIVE_ROOT = Path(
    "docs/plans/artifacts/"
    "ssl-lstm-q20-seed-b-terminal-neutra-validation-2026-08-07/r2/sequential"
)

TARGET_SIGNATURE = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
BASE_ADAPTER_SIGNATURE = "a8be6c212eb2a74ef926ccb9279871805949cae6e00c312a12525141638166f3"
Q = 20
CHAIN_COUNT = 4
PARAMETER_DIM = 4
EXPECTED_DRAWS = 1000
HORIZON = 10
REPLICATIONS = 2
CANARY_DRAWS = 32
SCALE_DRAWS = 2048
CALIBRATION_DRAWS = 16384
MATERIAL_DRAWS = 2048
BANDWIDTH_SUBSET_DRAWS = 128
BLOCK_LENGTH = 1
FEATURE_ALPHA = 0.03
MMD_ALPHA = 0.02
TOTAL_ALPHA = 0.05
MEAN_MARGIN = 0.15
LOG_VARIANCE_MARGIN = math.log(1.15)
SCALE_FLOOR = 1.0e-8
SKEW_COEFFICIENT = 0.35
MMD_TOLERANCES = (0.0005, 0.00075, 0.001, 0.00125, 0.0015, 0.00175, 0.002, 0.00225, 0.0025, 0.003)
RIDGE_LADDER = (0.0, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6)
CONDITION_NUMBER_MAX = 1.0e8
CHAIN_PAIR_SCHEDULE = ((0, 1), (2, 3))
FAMILY_ROLES = {
    "identical": ("equivalence", "PASS"),
    "negligible_mean": ("equivalence", "PASS"),
    "negligible_variance": ("equivalence", "PASS"),
    "material_mean": ("material", "MATERIAL_DIFFERENCE"),
    "material_variance": ("material", "MATERIAL_DIFFERENCE"),
    "shape_only_skew": ("material", "MATERIAL_DIFFERENCE"),
}
STAGE_CAPS = {
    "canary": 900.0,
    "scale": 900.0,
    "nominate": 3600.0,
    "validate": 10800.0,
    "material": 900.0,
    "audit": 900.0,
}


class RepairError(RuntimeError):
    """Raised when an experimental invariant or stage gate fails."""


def _abs(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha(path: Path) -> str:
    return hashlib.sha256(_abs(path).read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(_abs(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RepairError(f"expected a JSON object: {path}")
    return payload


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("ascii")
    if hasattr(value, "as_list"):
        return _safe(value.as_list())
    if hasattr(value, "numpy"):
        return _safe(value.numpy())
    if hasattr(value, "tolist"):
        return _safe(value.tolist())
    if hasattr(value, "item"):
        return _safe(value.item())
    return value


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    absolute = _abs(path)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if absolute.exists():
        raise RepairError(f"refusing to overwrite existing artifact: {path}")
    absolute.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )


def _git_commit() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _lane_seeds(root: int, *, arm: int = 0) -> tuple[tuple[int, int], ...]:
    return tuple((20260808, root + arm * 10 + lane) for lane in range(CHAIN_COUNT))


def _calibration_seeds(stage_root: int, replication: int, arm: int) -> tuple[tuple[int, int], ...]:
    return _lane_seeds(stage_root + 100 * replication, arm=arm)


def _configure_calibration_tensorflow() -> tuple[Any, Any]:
    # Calibration must remain independent of the NeuTra transport and archive.
    from ssl_lstm_q20_neutra_seed_b_terminal import configure_cpu_tensorflow

    tf = configure_cpu_tensorflow(threads=1)
    from bayesfilter.inference import predictive_equivalence as predictive

    return tf, predictive


def _true_parameter(tf: Any) -> Any:
    from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import PRIOR_CENTER

    return tf.convert_to_tensor(PRIOR_CENTER, tf.float64)


def _forecast(
    tf: Any,
    parameter: Any,
    seeds: tuple[tuple[int, int], ...],
    draws: int,
) -> Any:
    from bayesfilter.nonlinear.ssl_lstm_complexity_predictive_tf import (
        forecast_complexity_conditional_moments,
    )

    rows = []
    for seed in seeds:
        result = forecast_complexity_conditional_moments(
            tf.ensure_shape(parameter[tf.newaxis, :], [1, PARAMETER_DIM]),
            q=Q,
            seed=tf.constant(seed, tf.int32),
            replication_count=draws * REPLICATIONS,
        )
        observations = tf.reshape(
            tf.convert_to_tensor(result.observations, tf.float64),
            [draws, REPLICATIONS, HORIZON],
        )
        if not bool(tf.reduce_all(result.status)):
            raise RepairError(f"forecast target status failed for seed {seed}")
        if not bool(tf.reduce_all(tf.math.is_finite(observations))):
            raise RepairError(f"forecast produced nonfinite paths for seed {seed}")
        rows.append(observations)
    return tf.stack(rows, axis=0)


def _standardize(tf: Any, predictive: Any, paths: Any, scale: Mapping[str, Any]) -> Any:
    return predictive.standardize_forecast_paths(
        paths,
        tf.constant(scale["center"], tf.float64),
        tf.constant(scale["scale"], tf.float64),
        scale_floor=tf.constant(SCALE_FLOOR, tf.float64),
        jit_compile=True,
        allow_floor_use=False,
    )


def _margins(tf: Any) -> Any:
    return tf.concat(
        (
            tf.fill([HORIZON], tf.constant(MEAN_MARGIN, tf.float64)),
            tf.fill([HORIZON], tf.constant(LOG_VARIANCE_MARGIN, tf.float64)),
        ),
        axis=0,
    )


def _one_evidence(
    tf: Any,
    predictive: Any,
    left_standardized: Any,
    right_standardized: Any,
    scale: Mapping[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    left = predictive.mean_log_variance_influence(left_standardized, jit_compile=True)
    right = predictive.mean_log_variance_influence(right_standardized, jit_compile=True)
    estimate = left.feature_estimate - right.feature_estimate
    # Concatenating +2/-2 gives the correct covariance of two independent,
    # equal-size arm means when the covariance API pools over the leading axis.
    influence = tf.concat(
        (2.0 * left.influence_values, -2.0 * right.influence_values), axis=0
    )
    covariance = predictive.chain_batch_long_run_covariance(
        influence,
        block_length=BLOCK_LENGTH,
        ridge_ladder=RIDGE_LADDER,
        condition_number_max=CONDITION_NUMBER_MAX,
        jit_compile=True,
    )
    if not covariance.inference_admissible:
        raise RepairError("feature covariance is inadmissible")
    standard_error = tf.sqrt(tf.linalg.diag_part(covariance.regularized_covariance))
    feature_interval = predictive.simultaneous_feature_intervals(
        estimate,
        feature_alpha=FEATURE_ALPHA,
        method="bonferroni_studentized",
        standard_error=standard_error,
        jit_compile=True,
    )
    bandwidths = tf.constant(scale["bandwidths"], tf.float64)
    mmd = predictive.cross_chain_linear_mmd(
        left_standardized,
        right_standardized,
        bandwidths=bandwidths,
        mixture_weights=tf.fill([3], tf.constant(1.0 / 3.0, tf.float64)),
        chain_pair_schedule=tf.constant(CHAIN_PAIR_SCHEDULE, tf.int32),
        independent_arm_banks_verified=True,
        stationarity_verified=True,
        mixing_verified=True,
        jit_compile=True,
    )
    mmd_interval = predictive.cross_chain_mmd_upper_interval(
        mmd,
        mmd_alpha=MMD_ALPHA,
        block_length=BLOCK_LENGTH,
        jit_compile=True,
    )
    decision = predictive.classify_predictive_evidence(
        feature_interval,
        mmd_interval,
        margins=_margins(tf),
        mmd_tolerance=tf.constant(tolerance, tf.float64),
        total_alpha=TOTAL_ALPHA,
        feature_alpha=FEATURE_ALPHA,
        mmd_alpha=MMD_ALPHA,
    )
    return {
        "status": decision.status,
        "feature_status": decision.primary_interval_status,
        "mmd_status": decision.mmd_upper_bound_status,
        "hard_veto_codes": decision.hard_veto_codes,
        "feature_estimate": estimate,
        "feature_lower": feature_interval.lower,
        "feature_upper": feature_interval.upper,
        "feature_standard_error": feature_interval.standard_error,
        "feature_critical_value": feature_interval.critical_value,
        "mmd_estimate": mmd_interval.estimate,
        "mmd_lower": mmd_interval.lower,
        "mmd_upper": mmd_interval.upper,
        "mmd_standard_error": mmd_interval.standard_error,
        "mmd_critical_value": mmd_interval.critical_value,
        "covariance_condition_number": covariance.condition_number,
        "selected_ridge_multiplier": covariance.selected_ridge_multiplier,
        "block_length": covariance.block_length,
    }


def _gaussian_shape_banks(
    tf: Any,
    scale: Mapping[str, Any],
    *,
    stage_root: int,
    replication: int,
) -> tuple[Any, Any]:
    correlation_factor = tf.constant(scale["correlation_cholesky"], tf.float64)
    shape = [CHAIN_COUNT, CALIBRATION_DRAWS, REPLICATIONS, HORIZON]

    def bank(arm: int) -> Any:
        normal = tf.random.stateless_normal(
            shape,
            seed=tf.constant((20260808, stage_root + 100 * replication + arm), tf.int32),
            dtype=tf.float64,
        )
        return tf.einsum("cdri,hi->cdrh", normal, correlation_factor)

    left = bank(0)
    right_gaussian = bank(1)
    coefficient = tf.constant(SKEW_COEFFICIENT, tf.float64)
    right = (
        right_gaussian + coefficient * (tf.square(right_gaussian) - 1.0)
    ) / tf.sqrt(1.0 + 2.0 * tf.square(coefficient))
    center = tf.constant(scale["center"], tf.float64)
    horizon_scale = tf.constant(scale["scale"], tf.float64)
    return center + horizon_scale * left, center + horizon_scale * right


def _family_paths(
    tf: Any,
    scale: Mapping[str, Any],
    left_raw: Any,
    right_raw: Any,
    *,
    shape_stage_root: int,
    replication: int,
) -> dict[str, tuple[Any, Any]]:
    center = tf.constant(scale["center"], tf.float64)
    return {
        "identical": (left_raw, right_raw),
        "negligible_mean": (left_raw, right_raw + tf.constant(0.05, tf.float64)),
        "negligible_variance": (
            left_raw,
            center + tf.sqrt(tf.constant(1.05, tf.float64)) * (right_raw - center),
        ),
        "material_mean": (left_raw, right_raw + tf.constant(0.20, tf.float64)),
        "material_variance": (
            left_raw,
            center + tf.sqrt(tf.constant(1.25, tf.float64)) * (right_raw - center),
        ),
        "shape_only_skew": _gaussian_shape_banks(
            tf, scale, stage_root=shape_stage_root, replication=replication
        ),
    }


def _known_truth(tf: Any, scale: Mapping[str, Any], family: str) -> Any:
    zero = tf.zeros([HORIZON], tf.float64)
    if family == "negligible_mean":
        mean = -tf.constant(0.05, tf.float64) / tf.constant(scale["scale"], tf.float64)
        log_variance = zero
    elif family == "material_mean":
        mean = -tf.constant(0.20, tf.float64) / tf.constant(scale["scale"], tf.float64)
        log_variance = zero
    elif family == "negligible_variance":
        mean = zero
        log_variance = tf.fill([HORIZON], -tf.math.log(tf.constant(1.05, tf.float64)))
    elif family == "material_variance":
        mean = zero
        log_variance = tf.fill([HORIZON], -tf.math.log(tf.constant(1.25, tf.float64)))
    else:
        mean = zero
        log_variance = zero
    return tf.concat((mean, log_variance), axis=0)


def _covers_truth(tf: Any, evidence: Mapping[str, Any], truth: Any) -> bool:
    return bool(
        tf.reduce_all(
            tf.logical_and(
                tf.constant(evidence["feature_lower"], tf.float64) <= truth,
                truth <= tf.constant(evidence["feature_upper"], tf.float64),
            )
        )
    )


def _count_status(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    statuses = ("PASS", "MATERIAL_DIFFERENCE", "INCONCLUSIVE_UNDERPOWERED", "INVALID_HARD_VETO")
    return {status: sum(row["status"] == status for row in rows) for status in statuses}


def _candidate_passes(counts: Mapping[str, Mapping[str, int]], *, validation: bool) -> bool:
    threshold = 54 if validation else 16
    for family, (role, _expected) in FAMILY_ROLES.items():
        row = counts[family]
        if row["INVALID_HARD_VETO"] != 0:
            return False
        if role == "equivalence":
            if row["PASS"] < threshold or row["MATERIAL_DIFFERENCE"] > 1:
                return False
        elif row["MATERIAL_DIFFERENCE"] < threshold or row["PASS"] > 1:
            return False
    return True


def select_smallest_tolerance(candidate_rows: list[Mapping[str, Any]]) -> float | None:
    for row in sorted(candidate_rows, key=lambda item: float(item["tolerance"])):
        if _candidate_passes(row["counts"], validation=False):
            return float(row["tolerance"])
    return None


def _stage_receipt(
    path: Path,
    *,
    schema: str,
    required_status: str,
    bind_paths: tuple[Path, ...] = (),
) -> dict[str, Any]:
    if not _abs(path).is_file():
        raise RepairError(f"required prior-stage receipt is missing: {path}")
    payload = _read_json(path)
    if payload.get("schema") != schema or payload.get("status") != required_status:
        raise RepairError(f"prior-stage receipt did not pass: {path}")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise RepairError(f"prior-stage receipt lacks provenance: {path}")
    if provenance.get("plan_sha256") != _sha(PLAN):
        raise RepairError(f"prior-stage plan binding mismatch: {path}")
    if provenance.get("runner_sha256") != _sha(RUNNER):
        raise RepairError(f"prior-stage runner binding mismatch: {path}")
    for bound in bind_paths:
        key = f"{bound.stem}_sha256"
        if provenance.get(key) != _sha(bound):
            raise RepairError(f"prior-stage binding mismatch for {bound}")
    return payload


def _provenance(**extra: Any) -> dict[str, Any]:
    return {
        "plan": PLAN.as_posix(),
        "plan_sha256": _sha(PLAN),
        "runner": RUNNER.as_posix(),
        "runner_sha256": _sha(RUNNER),
        **extra,
    }


def _manifest(mode: str, started: float, cap_seconds: float, tf: Any) -> dict[str, Any]:
    return {
        "git_commit": _git_commit(),
        "git_dirty": bool(
            subprocess.run(
                ("git", "status", "--porcelain"),
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ),
        "command": " ".join(sys.argv),
        "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "mode": mode,
        "cpu_gpu_status": "CPU_ONLY_GPU_HIDDEN",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "jit_compile": True,
        "sample_wise_scalar_fallback_used": False,
        "iid_fixed_parameter_forecasts": True,
        "block_length": BLOCK_LENGTH,
        "wall_time_seconds": time.perf_counter() - started,
        "cap_seconds": cap_seconds,
        "cpu_only_reference_exception": True,
        "parameter_mixture_used": False,
    }


def _run_canary(output_root: Path, cap_seconds: float, started: float) -> dict[str, Any]:
    tf, predictive = _configure_calibration_tensorflow()
    true = _true_parameter(tf)
    left = _forecast(tf, true, _lane_seeds(110000, arm=0), CANARY_DRAWS)
    right = _forecast(tf, true, _lane_seeds(110000, arm=1), CANARY_DRAWS)
    center = tf.reduce_mean(tf.reshape(left, [-1, HORIZON]), axis=0)
    scale = tf.math.reduce_std(tf.reshape(left, [-1, HORIZON]), axis=0)
    standardized_left = predictive.standardize_forecast_paths(
        left,
        center,
        scale,
        scale_floor=tf.constant(SCALE_FLOOR, tf.float64),
        jit_compile=True,
        allow_floor_use=False,
    )
    standardized_right = predictive.standardize_forecast_paths(
        right,
        center,
        scale,
        scale_floor=tf.constant(SCALE_FLOOR, tf.float64),
        jit_compile=True,
        allow_floor_use=False,
    )
    distance = predictive.pooled_pairwise_distance_scale(
        standardized_left[:, :16], jit_compile=True
    )
    scale_payload = {
        "center": center,
        "scale": scale,
        "bandwidths": tf.constant((0.5, 1.0, 2.0), tf.float64) * distance.median_distance,
    }
    evidence = _one_evidence(
        tf, predictive, standardized_left, standardized_right, scale_payload, 0.01
    )
    status = "CANARY_PASSED" if evidence["status"] != "INVALID_HARD_VETO" else "INVALID_HARD_VETO"
    payload = {
        "schema": "bayesfilter.ssl_lstm.q20_predictive_equivalence_repair.canary.v1",
        "status": status,
        "evidence_status": evidence["status"],
        "path_shape": left.shape,
        "finite": bool(tf.reduce_all(tf.math.is_finite(left))),
        "standardization_applied": True,
        "block_length": BLOCK_LENGTH,
        "two_arm_influence_factors": (2.0, -2.0),
        "archive_or_transport_loaded": False,
        "provenance": _provenance(),
    }
    payload["run_manifest"] = _manifest("canary", started, cap_seconds, tf)
    return payload


def _run_scale(output_root: Path, cap_seconds: float, started: float) -> dict[str, Any]:
    tf, predictive = _configure_calibration_tensorflow()
    true = _true_parameter(tf)
    seeds = _lane_seeds(100000)
    raw = _forecast(tf, true, seeds, SCALE_DRAWS)
    flat = tf.reshape(raw, [-1, HORIZON])
    center = tf.reduce_mean(flat, axis=0)
    scale = tf.math.reduce_std(flat, axis=0)
    if not bool(tf.reduce_all(scale >= tf.constant(SCALE_FLOOR, tf.float64))):
        raise RepairError("scale bank would use the forbidden scale floor")
    standardized = predictive.standardize_forecast_paths(
        raw,
        center,
        scale,
        scale_floor=tf.constant(SCALE_FLOOR, tf.float64),
        jit_compile=True,
        allow_floor_use=False,
    )
    bandwidth_source = standardized[:, :BANDWIDTH_SUBSET_DRAWS]
    distance = predictive.pooled_pairwise_distance_scale(
        bandwidth_source, jit_compile=True
    )
    bandwidths = tf.constant((0.5, 1.0, 2.0), tf.float64) * distance.median_distance
    covariance = tf.matmul(flat - center, flat - center, transpose_a=True) / tf.cast(
        tf.shape(flat)[0], tf.float64
    )
    correlation = covariance / (scale[:, tf.newaxis] * scale[tf.newaxis, :])
    correlation = 0.5 * (correlation + tf.transpose(correlation))
    tf.debugging.assert_near(
        tf.linalg.diag_part(correlation),
        tf.ones([HORIZON], tf.float64),
        atol=tf.constant(1.0e-12, tf.float64),
        rtol=tf.constant(0.0, tf.float64),
        message="scale-bank correlation must have unit diagonal",
    )
    correlation_cholesky = tf.linalg.cholesky(correlation)
    payload = {
        "schema": "bayesfilter.ssl_lstm.q20_predictive_equivalence_repair.scale.v1",
        "status": "SCALE_PASSED",
        "center": center,
        "scale": scale,
        "scale_floor": SCALE_FLOOR,
        "scale_floor_used": False,
        "median_null_complete_path_distance": distance.median_distance,
        "bandwidths": bandwidths,
        "bandwidth_multipliers": (0.5, 1.0, 2.0),
        "bandwidth_source": "standardized_null_true_control_scale_bank_only",
        "bandwidth_subset_draws_per_lane": BANDWIDTH_SUBSET_DRAWS,
        "shifted_or_candidate_paths_used": False,
        "correlation": correlation,
        "correlation_cholesky": correlation_cholesky,
        "seed_lanes": seeds,
        "draws_per_lane": SCALE_DRAWS,
        "forecast_replications_per_draw": REPLICATIONS,
        "path_shape": raw.shape,
        "standardization_applied": True,
        "archive_or_transport_loaded": False,
        "provenance": _provenance(),
    }
    payload["run_manifest"] = _manifest("scale", started, cap_seconds, tf)
    return payload


def _calibration_replication(
    tf: Any,
    predictive: Any,
    scale: Mapping[str, Any],
    *,
    stage_root: int,
    shape_stage_root: int,
    replication: int,
    tolerances: tuple[float, ...],
) -> dict[float, dict[str, dict[str, Any]]]:
    true = _true_parameter(tf)
    left = _forecast(
        tf, true, _calibration_seeds(stage_root, replication, 0), CALIBRATION_DRAWS
    )
    right = _forecast(
        tf, true, _calibration_seeds(stage_root, replication, 1), CALIBRATION_DRAWS
    )
    families = _family_paths(
        tf,
        scale,
        left,
        right,
        shape_stage_root=shape_stage_root,
        replication=replication,
    )
    result: dict[float, dict[str, dict[str, Any]]] = {value: {} for value in tolerances}
    for family, (left_raw, right_raw) in families.items():
        left_standardized = _standardize(tf, predictive, left_raw, scale)
        right_standardized = _standardize(tf, predictive, right_raw, scale)
        # Intervals and MMD do not depend on the decision tolerance. Compute once.
        base = _one_evidence(
            tf,
            predictive,
            left_standardized,
            right_standardized,
            scale,
            tolerances[0],
        )
        for tolerance in tolerances:
            if tolerance == tolerances[0]:
                evidence = base
            else:
                # Reuse authenticated intervals by recomputing the cheap classifier
                # inside _one_evidence would repeat all statistics, so classify from
                # numeric interval boundaries with exactly the repository rules.
                lower = tf.constant(base["feature_lower"], tf.float64)
                upper = tf.constant(base["feature_upper"], tf.float64)
                margins = _margins(tf)
                feature_material = bool(tf.reduce_any(tf.logical_or(lower > margins, upper < -margins)))
                feature_inside = bool(tf.reduce_all(tf.logical_and(lower > -margins, upper < margins)))
                feature_status = "MATERIAL_DIFFERENCE" if feature_material else ("PASS" if feature_inside else "INCONCLUSIVE_UNDERPOWERED")
                mmd_lower = float(base["mmd_lower"])
                mmd_upper = float(base["mmd_upper"])
                mmd_status = "MATERIAL_DIFFERENCE" if mmd_lower > tolerance else ("PASS" if mmd_upper < tolerance else "INCONCLUSIVE_UNDERPOWERED")
                status = "MATERIAL_DIFFERENCE" if "MATERIAL_DIFFERENCE" in (feature_status, mmd_status) else ("PASS" if feature_status == mmd_status == "PASS" else "INCONCLUSIVE_UNDERPOWERED")
                evidence = {**base, "status": status, "feature_status": feature_status, "mmd_status": mmd_status}
            result[tolerance][family] = {
                **evidence,
                "known_truth": _known_truth(tf, scale, family),
                "known_truth_covered": _covers_truth(
                    tf, evidence, _known_truth(tf, scale, family)
                ),
            }
    return result


def _run_nominate(output_root: Path, cap_seconds: float, started: float) -> dict[str, Any]:
    scale_path = output_root / "scale.json"
    scale = _stage_receipt(
        scale_path,
        schema="bayesfilter.ssl_lstm.q20_predictive_equivalence_repair.scale.v1",
        required_status="SCALE_PASSED",
    )
    tf, predictive = _configure_calibration_tensorflow()
    by_tolerance = {
        tolerance: {family: [] for family in FAMILY_ROLES}
        for tolerance in MMD_TOLERANCES
    }
    for replication in range(20):
        rows = _calibration_replication(
            tf,
            predictive,
            scale,
            stage_root=900000,
            shape_stage_root=1100000,
            replication=replication,
            tolerances=MMD_TOLERANCES,
        )
        for tolerance, families in rows.items():
            for family, evidence in families.items():
                by_tolerance[tolerance][family].append(evidence)
    candidates = []
    for tolerance in MMD_TOLERANCES:
        family_rows = by_tolerance[tolerance]
        counts = {family: _count_status(rows) for family, rows in family_rows.items()}
        candidates.append(
            {
                "tolerance": tolerance,
                "counts": counts,
                "coverage_counts": {
                    family: sum(bool(row["known_truth_covered"]) for row in rows)
                    for family, rows in family_rows.items()
                },
                "rows": family_rows,
            }
        )
    selected = select_smallest_tolerance(candidates)
    status = "NOMINATION_PASSED" if selected is not None else "NOMINATION_FAILED"
    payload = {
        "schema": "bayesfilter.ssl_lstm.q20_predictive_equivalence_repair.nomination.v1",
        "status": status,
        "selected_mmd_tolerance": selected,
        "candidate_rows": candidates,
        "replication_count": 20,
        "draws_per_lane": CALIBRATION_DRAWS,
        "forecast_replications_per_draw": REPLICATIONS,
        "complete_decision_calibrated": True,
        "block_length": BLOCK_LENGTH,
        "provenance": _provenance(scale_sha256=_sha(scale_path)),
    }
    payload["run_manifest"] = _manifest("nominate", started, cap_seconds, tf)
    return payload


def _run_validate(output_root: Path, cap_seconds: float, started: float) -> dict[str, Any]:
    scale_path = output_root / "scale.json"
    nomination_path = output_root / "nominate.json"
    scale = _stage_receipt(
        scale_path,
        schema="bayesfilter.ssl_lstm.q20_predictive_equivalence_repair.scale.v1",
        required_status="SCALE_PASSED",
    )
    nomination = _stage_receipt(
        nomination_path,
        schema="bayesfilter.ssl_lstm.q20_predictive_equivalence_repair.nomination.v1",
        required_status="NOMINATION_PASSED",
        bind_paths=(scale_path,),
    )
    tolerance = float(nomination["selected_mmd_tolerance"])
    if tolerance not in MMD_TOLERANCES:
        raise RepairError("nominated tolerance is outside the frozen grid")
    tf, predictive = _configure_calibration_tensorflow()
    family_rows = {family: [] for family in FAMILY_ROLES}
    for replication in range(60):
        rows = _calibration_replication(
            tf,
            predictive,
            scale,
            stage_root=1300000,
            shape_stage_root=1500000,
            replication=replication,
            tolerances=(tolerance,),
        )[tolerance]
        for family, evidence in rows.items():
            family_rows[family].append(evidence)
    counts = {family: _count_status(rows) for family, rows in family_rows.items()}
    coverage = {
        family: sum(bool(row["known_truth_covered"]) for row in rows)
        for family, rows in family_rows.items()
    }
    decision_gate = _candidate_passes(counts, validation=True)
    required_coverage = all(
        coverage[family] >= 54
        for family in ("identical", "negligible_mean", "material_mean", "shape_only_skew")
    )
    invalid_count = sum(
        counts[family]["INVALID_HARD_VETO"] for family in FAMILY_ROLES
    )
    passed = decision_gate and required_coverage and invalid_count == 0
    payload = {
        "schema": "bayesfilter.ssl_lstm.q20_predictive_equivalence_repair.validation.v1",
        "status": "VALIDATION_PASSED" if passed else "VALIDATION_FAILED",
        "selected_mmd_tolerance": tolerance,
        "counts": counts,
        "coverage_counts": coverage,
        "decision_gate_passed": decision_gate,
        "required_coverage_passed": required_coverage,
        "variance_family_coverage_explanatory_only": True,
        "rows": family_rows,
        "replication_count": 60,
        "draws_per_lane": CALIBRATION_DRAWS,
        "forecast_replications_per_draw": REPLICATIONS,
        "complete_decision_calibrated": True,
        "block_length": BLOCK_LENGTH,
        "provenance": _provenance(
            scale_sha256=_sha(scale_path), nominate_sha256=_sha(nomination_path)
        ),
    }
    payload["run_manifest"] = _manifest("validate", started, cap_seconds, tf)
    return payload


def _load_retained(tf: Any) -> tuple[Any, dict[str, Any]]:
    summary_path = ARCHIVE_ROOT / "summary.json"
    summary = _read_json(summary_path)
    if summary.get("status") != "SEQUENTIAL_SCREEN_PASSED" or summary.get("passed") is not True:
        raise RepairError("seed-B sequential archive is not admitted")
    if int(summary.get("retained_results_per_chain", -1)) != EXPECTED_DRAWS:
        raise RepairError("unexpected retained draw count")
    manifest_path = ARCHIVE_ROOT / "archive/seed-b-terminal-manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("warmup_excluded_from_posterior") is not True:
        raise RepairError("warm-up exclusion is not authenticated")
    chunks = []
    receipts = []
    for index in (0, 1):
        receipt_path = ARCHIVE_ROOT / "archive/retained" / f"seed-b-terminal-retained-{index:03d}-receipt.json"
        receipt = _read_json(receipt_path)
        sample = receipt.get("sample_receipt")
        if not isinstance(sample, Mapping):
            raise RepairError("retained receipt lacks sample descriptor")
        sample_path = Path(str(sample.get("path")))
        if _sha(sample_path) != sample.get("sha256"):
            raise RepairError(f"retained tensor hash mismatch: {index}")
        tensor = tf.io.parse_tensor(_abs(sample_path).read_bytes(), out_type=tf.float64)
        if tuple(tensor.shape) != (500, CHAIN_COUNT, PARAMETER_DIM):
            raise RepairError("retained tensor shape mismatch")
        if not bool(tf.reduce_all(tf.math.is_finite(tensor))):
            raise RepairError("retained tensor is nonfinite")
        chunks.append(tensor)
        receipts.append(
            {
                "receipt": receipt_path.as_posix(),
                "receipt_sha256": _sha(receipt_path),
                "sample": sample,
            }
        )
    samples = tf.transpose(tf.concat(chunks, axis=0), (1, 0, 2))
    return samples, {
        "summary_sha256": _sha(summary_path),
        "archive_manifest_sha256": _sha(manifest_path),
        "retained_receipts": receipts,
        "shape": samples.shape,
    }


def _parameters(tf: Any, transport: Any, provenance: Mapping[str, Any]) -> tuple[Any, Any]:
    if provenance.get("target_signature") != TARGET_SIGNATURE:
        raise RepairError("target signature mismatch")
    if provenance.get("target_adapter_signature") != BASE_ADAPTER_SIGNATURE:
        raise RepairError("target adapter signature mismatch")
    retained, archive = _load_retained(tf)
    theta = tf.convert_to_tensor(
        transport.forward_z_to_theta_batch(tf.reshape(retained, (-1, PARAMETER_DIM))),
        tf.float64,
    )
    if tuple(theta.shape) != (CHAIN_COUNT * EXPECTED_DRAWS, PARAMETER_DIM):
        raise RepairError("mapped retained draw shape mismatch")
    if not bool(tf.reduce_all(tf.math.is_finite(theta))):
        raise RepairError("mapped retained draws are nonfinite")
    return tf.reduce_mean(theta, axis=0), archive


def _raw_summary(tf: Any, paths: Any) -> dict[str, Any]:
    flat = tf.reshape(paths, [-1, HORIZON])
    return {
        "mean": tf.reduce_mean(flat, axis=0),
        "variance": tf.math.reduce_variance(flat, axis=0),
        "path_count": tf.shape(flat)[0],
    }


def _run_material_like(
    mode: str, output_root: Path, cap_seconds: float, started: float
) -> dict[str, Any]:
    scale_path = output_root / "scale.json"
    nomination_path = output_root / "nominate.json"
    validation_path = output_root / "validate.json"
    material_path = output_root / "material.json"
    scale = _stage_receipt(
        scale_path,
        schema="bayesfilter.ssl_lstm.q20_predictive_equivalence_repair.scale.v1",
        required_status="SCALE_PASSED",
    )
    validation = _stage_receipt(
        validation_path,
        schema="bayesfilter.ssl_lstm.q20_predictive_equivalence_repair.validation.v1",
        required_status="VALIDATION_PASSED",
        bind_paths=(scale_path, nomination_path),
    )
    if mode == "audit":
        _stage_receipt(
            material_path,
            schema="bayesfilter.ssl_lstm.q20_predictive_equivalence_repair.material.v1",
            required_status="PASS",
            bind_paths=(scale_path, nomination_path, validation_path),
        )
    from ssl_lstm_q20_neutra_seed_b_terminal import build_seed_b_terminal

    _, transport, transport_provenance = build_seed_b_terminal(
        threads=1,
        evidence_path=PLAN.as_posix(),
        target_scope_suffix=f"predictive_equivalence_repair_{mode}",
    )
    import tensorflow as tf
    from bayesfilter.inference import predictive_equivalence as predictive

    posterior_mean, archive = _parameters(tf, transport, transport_provenance)
    true = _true_parameter(tf)
    seed_root = 700000 if mode == "material" else 800000
    true_paths = _forecast(
        tf, true, _lane_seeds(seed_root, arm=0), MATERIAL_DRAWS
    )
    candidate_paths = _forecast(
        tf, posterior_mean, _lane_seeds(seed_root, arm=1), MATERIAL_DRAWS
    )
    true_standardized = _standardize(tf, predictive, true_paths, scale)
    candidate_standardized = _standardize(tf, predictive, candidate_paths, scale)
    tolerance = float(validation["selected_mmd_tolerance"])
    evidence = _one_evidence(
        tf,
        predictive,
        candidate_standardized,
        true_standardized,
        scale,
        tolerance,
    )
    schema_mode = "material" if mode == "material" else "audit"
    payload = {
        "schema": f"bayesfilter.ssl_lstm.q20_predictive_equivalence_repair.{schema_mode}.v1",
        "status": evidence["status"],
        "decision": evidence,
        "parameter_summaries": {"true": true, "posterior_mean": posterior_mean},
        "archive": archive,
        "target_signature": TARGET_SIGNATURE,
        "base_adapter_signature": BASE_ADAPTER_SIGNATURE,
        "raw_summaries": {
            "true": _raw_summary(tf, true_paths),
            "posterior_mean": _raw_summary(tf, candidate_paths),
        },
        "contract": {
            "parameter_mixture_used": False,
            "parameter_rows_per_arm": 1,
            "draws_per_lane": MATERIAL_DRAWS,
            "forecast_replications_per_draw": REPLICATIONS,
            "lanes": CHAIN_COUNT,
            "iid_fixed_parameter_forecasts": True,
            "independent_seed_banks": True,
            "standardization_applied": True,
            "mmd_tolerance": tolerance,
            "block_length": BLOCK_LENGTH,
        },
        "provenance": _provenance(
            scale_sha256=_sha(scale_path),
            nominate_sha256=_sha(nomination_path),
            validate_sha256=_sha(validation_path),
            **({"material_sha256": _sha(material_path)} if mode == "audit" else {}),
        ),
    }
    payload["run_manifest"] = _manifest(mode, started, cap_seconds, tf)
    return payload


def run(mode: str, output_root: Path, cap_seconds: float) -> dict[str, Any]:
    if mode not in STAGE_CAPS:
        raise RepairError(f"unsupported mode: {mode}")
    started = time.perf_counter()
    if mode == "canary":
        payload = _run_canary(output_root, cap_seconds, started)
    elif mode == "scale":
        payload = _run_scale(output_root, cap_seconds, started)
    elif mode == "nominate":
        payload = _run_nominate(output_root, cap_seconds, started)
    elif mode == "validate":
        payload = _run_validate(output_root, cap_seconds, started)
    else:
        payload = _run_material_like(mode, output_root, cap_seconds, started)
    elapsed = time.perf_counter() - started
    if elapsed > cap_seconds:
        raise RepairError(f"{mode} wall cap exceeded: {elapsed:.3f} > {cap_seconds:.3f}")
    _write(output_root / f"{mode}.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=tuple(STAGE_CAPS), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--cap-seconds", type=float)
    args = parser.parse_args(argv)
    cap = STAGE_CAPS[args.mode] if args.cap_seconds is None else args.cap_seconds
    payload = run(args.mode, args.output_root, cap)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "status": payload["status"],
                "wall_time_seconds": payload["run_manifest"]["wall_time_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
