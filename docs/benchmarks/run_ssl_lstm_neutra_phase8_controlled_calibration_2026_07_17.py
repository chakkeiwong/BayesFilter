#!/usr/bin/env python3
"""Controlled null/power calibration for the Phase 8 predictive design."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import tensorflow as tf
import tensorflow_probability as tfp


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bayesfilter.inference.predictive_equivalence as predictive  # noqa: E402


PLAN_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-neutra-phase-8-predictive-design-refresh-"
    "plan-2026-07-17.md"
)
SCRIPT_PATH = Path(__file__).resolve().relative_to(ROOT)
PILOT_RECEIPT_PATH = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/"
    "target-pilot-repair-03.json"
)
PILOT_RECEIPT_SHA256 = (
    "5ae511c248e222edf14660c91c4a48412706c6f452298ebe5e144bbe8f01c098"
)
SMOKE_RECEIPT_PATH = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/"
    "controlled-calibration-smoke-repair-01.json"
)
SMOKE_RECEIPT_SHA256 = (
    "35201abd756a39a0c16a61477ad58b58c77d11ba00b3ef362484c42828f75c66"
)
SMOKE_RUNNER_SHA256 = (
    "f4315027c230e52bbd7e99b86e12e1379c98f7cd713d9d3de947331c3bfbf171"
)
PREDICTIVE_SOURCE = Path("bayesfilter/inference/predictive_equivalence.py")
CHAIN_COUNT = 4
DRAW_COUNT = 448
REPLICATION_COUNT = 2
HORIZON = 10
BLOCK_LENGTH = 16
FEATURE_ALPHA = 0.03
MMD_ALPHA = 0.02
TOTAL_ALPHA = 0.05
MEAN_MARGIN = 0.15
LOG_VARIANCE_MARGIN = math.log(1.15)
MMD_TOLERANCES = (0.005, 0.01, 0.02, 0.04, 0.08, 0.16)
RIDGE_LADDER = (0.0, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6)
CONDITION_NUMBER_MAX = 1.0e8
NOMINATION_COUNT = 20
VALIDATION_COUNT = 60
NOMINATION_SEED = (14001, 14002)
VALIDATION_SEED = (15001, 15002)
CHAIN_PAIR_SCHEDULE = ((0, 1), (2, 3))


@dataclass(frozen=True)
class Family:
    name: str
    role: Literal["equivalence", "material", "explanatory"]
    phi: float = 0.6
    horizon_rho: float = 0.7
    mean_shift: float = 0.0
    variance_ratio: float = 1.0
    local_horizon: int | None = None
    skew: float = 0.0


FAMILIES = (
    Family("iid_null", "equivalence", phi=0.0),
    Family("ar_null", "equivalence"),
    Family("trueeq_mean_persistent_0p05", "equivalence", mean_shift=0.05),
    Family("trueeq_variance_persistent_1p05", "equivalence", variance_ratio=1.05),
    Family("material_mean_persistent_pos0p20", "material", mean_shift=0.20),
    Family("material_mean_persistent_neg0p20", "material", mean_shift=-0.20),
    Family("material_mean_local_h1_pos0p20", "material", mean_shift=0.20, local_horizon=0),
    Family("material_mean_local_h1_neg0p20", "material", mean_shift=-0.20, local_horizon=0),
    Family("material_variance_persistent_1p25", "material", variance_ratio=1.25),
    Family("material_variance_persistent_0p80", "material", variance_ratio=0.80),
    Family("material_variance_local_h1_1p25", "material", variance_ratio=1.25, local_horizon=0),
    Family("skew_explanatory", "explanatory", skew=0.35),
    Family("dependence_explanatory", "explanatory", horizon_rho=0.9),
)


class CalibrationError(RuntimeError):
    """Raised when controlled calibration fails closed."""


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(_absolute(path).read_bytes()).hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise CalibrationError(f"duplicate JSON key {key!r}: {path}")
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise CalibrationError(f"nonfinite JSON constant {value!r}: {path}")

    value = json.loads(
        _absolute(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=reject,
    )
    if not isinstance(value, dict):
        raise CalibrationError(f"expected JSON object: {path}")
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("ascii")
    if hasattr(value, "numpy"):
        return _json_safe(value.numpy())
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


def _canonical(payload: Any) -> bytes:
    return (
        json.dumps(
            _json_safe(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _write(path: Path, payload: dict[str, Any]) -> None:
    absolute = _absolute(path)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if absolute.exists():
        raise CalibrationError(f"refusing to overwrite receipt: {path}")
    absolute.write_bytes(_canonical(payload))


def _git(*arguments: str) -> str:
    environment = dict(os.environ)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _status(value: tf.Tensor) -> str:
    raw = value.numpy()
    return raw.decode("ascii") if isinstance(raw, bytes) else str(raw)


def _trace_count(program: Any) -> int | None:
    method = getattr(program, "experimental_get_tracing_count", None)
    return None if method is None else int(method())


def _require_gpu(*tensors: tf.Tensor, surface: str) -> None:
    devices = [str(tensor.device) for tensor in tensors]
    if not devices or any("GPU:" not in device for device in devices):
        raise CalibrationError(f"{surface} outputs are not GPU resident: {devices}")


def _validate_pilot() -> dict[str, Any]:
    if _sha256(PILOT_RECEIPT_PATH) != PILOT_RECEIPT_SHA256:
        raise CalibrationError("target-pilot receipt identity drift")
    pilot = _strict_json(PILOT_RECEIPT_PATH)
    if (
        pilot.get("status") != "PASSED"
        or pilot.get("decision")
        != "PHASE8_TARGET_PILOT_PASSED_CONTROL_CALIBRATION_REQUIRED"
        or pilot.get("split_contract", {}).get("confirmation_forecast_bank_opened")
        is not False
        or pilot.get("split_contract", {}).get("g_h_predictive_difference_computed")
        is not False
    ):
        raise CalibrationError("target-pilot decision or leakage contract drift")
    return pilot


def _validate_smoke() -> dict[str, Any]:
    if _sha256(SMOKE_RECEIPT_PATH) != SMOKE_RECEIPT_SHA256:
        raise CalibrationError("controlled-calibration smoke receipt identity drift")
    smoke = _strict_json(SMOKE_RECEIPT_PATH)
    configuration = smoke.get("configuration", {})
    manifest = smoke.get("run_manifest", {})
    if (
        smoke.get("status") != "PASSED_EXECUTION"
        or smoke.get("decision")
        != "PHASE8_CONTROLLED_CALIBRATION_SMOKE_PASSED_NOMINATION_REQUIRED"
        or smoke.get("mode") != "smoke"
        or configuration.get("selected_mmd_tolerance") is not None
        or configuration.get("shape_per_arm")
        != [CHAIN_COUNT, DRAW_COUNT, REPLICATION_COUNT, HORIZON]
        or smoke.get("validation_exact_binomial_bounds") is not None
        or [row.get("name") for row in smoke.get("families", [])]
        != ["iid_null", "material_mean_persistent_pos0p20"]
        or set(smoke.get("compile_trace_counts", {}).values()) != {1}
        or manifest.get("cuda_visible_devices") != "1"
        or manifest.get("jit_compile") is not True
        or manifest.get("trust_basis")
        != "owner_designated_managed_session_visible_gpu_trusted"
        or smoke.get("pilot_binding", {}).get("sha256") != PILOT_RECEIPT_SHA256
        or smoke.get("source_bindings", {}).get("runner", {}).get("sha256")
        != SMOKE_RUNNER_SHA256
    ):
        raise CalibrationError("controlled-calibration smoke contract drift")
    return smoke


def _fold(seed: tf.Tensor, value: int) -> tf.Tensor:
    return tf.random.experimental.stateless_fold_in(
        seed, tf.constant(value, tf.int32), alg="philox"
    )


@tf.function(jit_compile=True, autograph=False)
def _gaussian_paths(seed: tf.Tensor, phi: tf.Tensor, rho: tf.Tensor) -> tf.Tensor:
    index = tf.range(HORIZON, dtype=tf.float64)
    covariance = tf.pow(rho, tf.abs(index[:, None] - index[None, :]))
    factor = tf.linalg.cholesky(covariance)
    draw_noise = tf.random.stateless_normal(
        [CHAIN_COUNT, DRAW_COUNT, HORIZON],
        _fold(seed, 1),
        dtype=tf.float64,
        alg="philox",
    ) @ tf.transpose(factor)
    initial = tf.random.stateless_normal(
        [CHAIN_COUNT, HORIZON],
        _fold(seed, 2),
        dtype=tf.float64,
        alg="philox",
    ) @ tf.transpose(factor)
    innovations = tf.transpose(draw_noise[:, 1:, :], [1, 0, 2])
    scanned = tf.scan(
        lambda previous, innovation: phi * previous
        + tf.sqrt(1.0 - tf.square(phi)) * innovation,
        innovations,
        initializer=initial,
    )
    clusters = tf.concat(
        (initial[:, None, :], tf.transpose(scanned, [1, 0, 2])), axis=1
    )
    replication_noise = tf.random.stateless_normal(
        [CHAIN_COUNT, DRAW_COUNT, REPLICATION_COUNT, HORIZON],
        _fold(seed, 3),
        dtype=tf.float64,
        alg="philox",
    ) @ tf.transpose(factor)
    return tf.sqrt(tf.constant(0.65, tf.float64)) * clusters[:, :, None, :] + tf.sqrt(
        tf.constant(0.35, tf.float64)
    ) * replication_noise


def _family_paths(root_seed: tuple[int, int], family_index: int, replication: int) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    seed = tf.constant(root_seed, tf.int32)
    seed = _fold(_fold(seed, family_index), replication)
    left = _gaussian_paths(_fold(seed, 11), tf.constant(FAMILIES[family_index].phi, tf.float64), tf.constant(0.7, tf.float64))
    family = FAMILIES[family_index]
    right = _gaussian_paths(
        _fold(seed, 22),
        tf.constant(family.phi, tf.float64),
        tf.constant(family.horizon_rho, tf.float64),
    )
    shift = tf.zeros([HORIZON], tf.float64)
    log_ratio = tf.zeros([HORIZON], tf.float64)
    if family.mean_shift:
        if family.local_horizon is None:
            shift = tf.fill([HORIZON], tf.constant(family.mean_shift, tf.float64))
        else:
            shift = tf.tensor_scatter_nd_update(
                shift, [[family.local_horizon]], [family.mean_shift]
            )
        right = right + shift
    if family.variance_ratio != 1.0:
        if family.local_horizon is None:
            ratios = tf.fill([HORIZON], tf.constant(family.variance_ratio, tf.float64))
        else:
            ratios = tf.ones([HORIZON], tf.float64)
            ratios = tf.tensor_scatter_nd_update(
                ratios, [[family.local_horizon]], [family.variance_ratio]
            )
        right = right * tf.sqrt(ratios)
        log_ratio = tf.math.log(ratios)
    if family.skew:
        coefficient = tf.constant(family.skew, tf.float64)
        right = (right + coefficient * (tf.square(right) - 1.0)) / tf.sqrt(
            1.0 + 2.0 * tf.square(coefficient)
        )
    truth = tf.concat((-shift, -log_ratio), axis=0)
    return left, right, truth


def _one_evidence(
    left: tf.Tensor,
    right: tf.Tensor,
    truth: tf.Tensor,
    *,
    bandwidths: tf.Tensor,
) -> dict[str, Any]:
    left_features = predictive.mean_log_variance_influence(left)
    right_features = predictive.mean_log_variance_influence(right)
    estimate = left_features.feature_estimate - right_features.feature_estimate
    influences = tf.concat(
        (2.0 * left_features.influence_values, -2.0 * right_features.influence_values),
        axis=0,
    )
    covariance = predictive.chain_batch_long_run_covariance(
        influences,
        block_length=BLOCK_LENGTH,
        ridge_ladder=RIDGE_LADDER,
        condition_number_max=CONDITION_NUMBER_MAX,
    )
    if not covariance.inference_admissible:
        raise CalibrationError("controlled feature covariance is inadmissible")
    standard_error = tf.sqrt(tf.linalg.diag_part(covariance.regularized_covariance))
    feature_interval = predictive.simultaneous_feature_intervals(
        estimate,
        feature_alpha=FEATURE_ALPHA,
        method="bonferroni_studentized",
        standard_error=standard_error,
    )
    mmd = predictive.cross_chain_linear_mmd(
        left,
        right,
        bandwidths=bandwidths,
        mixture_weights=tf.fill([5], tf.constant(0.2, tf.float64)),
        chain_pair_schedule=tf.constant(CHAIN_PAIR_SCHEDULE, tf.int32),
        independent_arm_banks_verified=True,
        stationarity_verified=True,
        mixing_verified=True,
    )
    mmd_interval = predictive.cross_chain_mmd_upper_interval(
        mmd, mmd_alpha=MMD_ALPHA, block_length=BLOCK_LENGTH
    )
    if not mmd_interval.inference_admissible:
        raise CalibrationError("controlled MMD interval is inadmissible")
    _require_gpu(
        estimate,
        covariance.regularized_covariance,
        feature_interval.lower,
        feature_interval.upper,
        mmd.kernel_contrast_sequence,
        mmd_interval.lower,
        mmd_interval.upper,
        surface="controlled calibration",
    )
    coverage = bool(
        tf.reduce_all(
            tf.logical_and(feature_interval.lower <= truth, truth <= feature_interval.upper)
        )
    )
    margins = _feature_margins()
    decisions = {}
    for tolerance in MMD_TOLERANCES:
        decision = predictive.classify_predictive_evidence(
            feature_interval,
            mmd_interval,
            margins=margins,
            mmd_tolerance=tf.constant(tolerance, tf.float64),
            total_alpha=TOTAL_ALPHA,
            feature_alpha=FEATURE_ALPHA,
            mmd_alpha=MMD_ALPHA,
        )
        if decision.status == "INVALID_HARD_VETO":
            raise CalibrationError(f"controlled decision hard veto: {decision.hard_veto_codes}")
        decisions[str(tolerance)] = decision.status
    return {
        "coverage": coverage,
        "feature_status_by_tolerance": decisions,
        "feature_estimate_max_abs": float(tf.reduce_max(tf.abs(estimate))),
        "feature_interval_max_width": float(
            tf.reduce_max(feature_interval.upper - feature_interval.lower)
        ),
        "mmd_estimate": float(mmd.squared_mmd_linear),
        "mmd_lower": float(mmd_interval.lower),
        "mmd_upper": float(mmd_interval.upper),
        "ridge_multiplier": float(covariance.selected_ridge_multiplier),
        "condition_number": float(covariance.condition_number),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for tolerance in MMD_TOLERANCES:
        key = str(tolerance)
        statuses = [row["feature_status_by_tolerance"][key] for row in rows]
        result[key] = {
            "pass_count": statuses.count("PASS"),
            "material_difference_count": statuses.count("MATERIAL_DIFFERENCE"),
            "inconclusive_count": statuses.count("INCONCLUSIVE_UNDERPOWERED"),
        }
    return {
        "replication_count": len(rows),
        "coverage_count": sum(row["coverage"] for row in rows),
        "by_tolerance": result,
        "max_interval_width": max(row["feature_interval_max_width"] for row in rows),
        "max_condition_number": max(row["condition_number"] for row in rows),
    }


def _nomination_pass(families: dict[str, Any], tolerance: float) -> bool:
    key = str(tolerance)
    for family in FAMILIES:
        if family.role == "explanatory":
            continue
        row = families[family.name]
        if row["replication_count"] != NOMINATION_COUNT:
            return False
        if row["coverage_count"] < 18:
            return False
        counts = row["by_tolerance"][key]
        if family.role == "equivalence":
            if counts["pass_count"] < 16 or counts["material_difference_count"] > 1:
                return False
        elif counts["pass_count"] > 1 or counts["material_difference_count"] < 16:
            return False
    return True


def _viable_nomination_tolerances(
    families: dict[str, Any], completed_count: int
) -> tuple[float, ...]:
    if not 0 <= completed_count <= NOMINATION_COUNT:
        raise CalibrationError("invalid nomination replication count")
    remaining = NOMINATION_COUNT - completed_count
    viable: list[float] = []
    for tolerance in MMD_TOLERANCES:
        key = str(tolerance)
        tolerance_viable = True
        for family in FAMILIES:
            if family.role == "explanatory":
                continue
            row = families[family.name]
            counts = row["by_tolerance"][key]
            if row["coverage_count"] + remaining < 18:
                tolerance_viable = False
            elif family.role == "equivalence" and (
                counts["pass_count"] + remaining < 16
                or counts["material_difference_count"] > 1
            ):
                tolerance_viable = False
            elif family.role == "material" and (
                counts["material_difference_count"] + remaining < 16
                or counts["pass_count"] > 1
            ):
                tolerance_viable = False
            if not tolerance_viable:
                break
        if tolerance_viable:
            viable.append(tolerance)
    return tuple(viable)


def _exact_bounds(successes: int, count: int) -> tuple[float, float]:
    alpha = tf.constant(0.05, tf.float64)
    lower = 0.0 if successes == 0 else float(
        tfp.distributions.Beta(
            tf.constant(float(successes), tf.float64),
            tf.constant(float(count - successes + 1), tf.float64),
        ).quantile(alpha)
    )
    upper = 1.0 if successes == count else float(
        tfp.distributions.Beta(
            tf.constant(float(successes + 1), tf.float64),
            tf.constant(float(count - successes), tf.float64),
        ).quantile(1.0 - alpha)
    )
    return lower, upper


def _validation_bounds(aggregates: dict[str, Any], tolerance: float) -> dict[str, Any]:
    key = str(tolerance)
    result: dict[str, Any] = {}
    for family in FAMILIES:
        row = aggregates[family.name]
        count = row["replication_count"]
        counts = row["by_tolerance"][key]
        coverage_lower, coverage_upper = _exact_bounds(row["coverage_count"], count)
        pass_lower, pass_upper = _exact_bounds(counts["pass_count"], count)
        material_lower, material_upper = _exact_bounds(
            counts["material_difference_count"], count
        )
        result[family.name] = {
            "role": family.role,
            "coverage": {"lower": coverage_lower, "upper": coverage_upper},
            "pass": {"lower": pass_lower, "upper": pass_upper},
            "material_difference": {
                "lower": material_lower,
                "upper": material_upper,
            },
        }
    return result


def _feature_margins() -> tf.Tensor:
    return tf.concat(
        (
            tf.fill([HORIZON], tf.constant(MEAN_MARGIN, tf.float64)),
            tf.fill([HORIZON], tf.constant(LOG_VARIANCE_MARGIN, tf.float64)),
        ),
        axis=0,
    )


def run(*, mode: str, output: Path, wall_cap_seconds: float, selected_tolerance: float | None) -> dict[str, Any]:
    if mode not in {"smoke", "nomination", "validation"}:
        raise CalibrationError("unknown calibration mode")
    if not math.isfinite(wall_cap_seconds) or wall_cap_seconds <= 0.0:
        raise CalibrationError("wall cap must be positive and finite")
    if mode == "validation":
        raise CalibrationError(
            "validation remains closed until the nomination receipt and selected "
            "tolerance are hard-bound in source"
        )
    pilot = _validate_pilot()
    smoke = _validate_smoke() if mode == "nomination" else None
    pooled = pilot["pooled_calibration"]
    bandwidths = tf.constant(pooled["bandwidth_candidates"], tf.float64)
    if tuple(bandwidths.shape) != (5,):
        raise CalibrationError("pilot bandwidth candidate shape drift")
    count = 1 if mode == "smoke" else (
        NOMINATION_COUNT if mode == "nomination" else VALIDATION_COUNT
    )
    root_seed = NOMINATION_SEED if mode in {"smoke", "nomination"} else VALIDATION_SEED
    active_family_indices = (
        (0, 4) if mode == "smoke" else tuple(range(len(FAMILIES)))
    )
    started_at = _now()
    started = time.perf_counter()
    family_rows: dict[str, list[dict[str, Any]]] = {
        FAMILIES[index].name: [] for index in active_family_indices
    }
    sequential_stop_reason = None
    viable_tolerances_after_stop: tuple[float, ...] | None = None
    for replication in range(count):
        for family_index in active_family_indices:
            family = FAMILIES[family_index]
            left, right, truth = _family_paths(root_seed, family_index, replication)
            _require_gpu(left, right, truth, surface="controlled family generation")
            family_rows[family.name].append(
                _one_evidence(left, right, truth, bandwidths=bandwidths)
            )
        if time.perf_counter() - started > wall_cap_seconds:
            raise CalibrationError("controlled calibration wall cap exceeded")
        if mode == "nomination":
            partial = {name: _aggregate(rows) for name, rows in family_rows.items()}
            viable = _viable_nomination_tolerances(partial, replication + 1)
            print(
                "NOMINATION_PROGRESS "
                + json.dumps(
                    {
                        "completed_replications": replication + 1,
                        "elapsed_seconds": time.perf_counter() - started,
                        "viable_tolerance_count": len(viable),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if replication + 1 < count and not viable:
                sequential_stop_reason = (
                    "all tolerance candidates are unable to meet the frozen "
                    "20-replication nomination thresholds"
                )
                viable_tolerances_after_stop = viable
                break
    aggregates = {name: _aggregate(rows) for name, rows in family_rows.items()}
    executed_count = len(next(iter(family_rows.values())))
    nominated = None
    validation_bounds = None
    if mode == "smoke":
        decision = "PHASE8_CONTROLLED_CALIBRATION_SMOKE_PASSED_NOMINATION_REQUIRED"
    elif mode == "nomination":
        nominated = next(
            (tolerance for tolerance in MMD_TOLERANCES if _nomination_pass(aggregates, tolerance)),
            None,
        )
        decision = (
            "PHASE8_CONTROLLED_NOMINATION_PASSED_FRESH_VALIDATION_REQUIRED"
            if nominated is not None
            else "PHASE8_CONTROLLED_NOMINATION_UNDERPOWERED_REPAIR_REQUIRED"
        )
    else:
        nominated = selected_tolerance
        key = str(selected_tolerance)
        validation_bounds = _validation_bounds(aggregates, selected_tolerance)
        passed = True
        for family in FAMILIES:
            if family.role == "explanatory":
                continue
            row = aggregates[family.name]
            coverage_lower = validation_bounds[family.name]["coverage"]["lower"]
            counts = row["by_tolerance"][key]
            if coverage_lower < 0.85:
                passed = False
            if family.role == "equivalence":
                pass_lower = validation_bounds[family.name]["pass"]["lower"]
                material_upper = validation_bounds[family.name]["material_difference"]["upper"]
                passed = passed and pass_lower >= 0.70 and material_upper <= 0.10
            else:
                pass_upper = validation_bounds[family.name]["pass"]["upper"]
                material_lower = validation_bounds[family.name]["material_difference"]["lower"]
                passed = passed and pass_upper <= 0.10 and material_lower >= 0.70
        decision = (
            "PHASE8_CONTROLLED_VALIDATION_PASSED_DESIGN_FREEZE_REQUIRED"
            if passed
            else "PHASE8_CONTROLLED_VALIDATION_FAILED_NEW_NOMINATION_REQUIRED"
        )
    wall_time = time.perf_counter() - started
    trace_counts = {
        "gaussian_paths": _trace_count(_gaussian_paths),
        "mean_log_variance_influence": _trace_count(
            predictive._mean_log_variance_influence_xla
        ),
        "long_run_covariance": _trace_count(predictive._long_run_covariance_xla),
        "bonferroni_interval": _trace_count(predictive._bonferroni_interval_xla),
        "cluster_kernel": _trace_count(predictive._cluster_kernel_xla),
        "mmd_block_interval": _trace_count(predictive._mmd_block_interval_xla),
    }
    if any(value != 1 for value in trace_counts.values()):
        raise CalibrationError(f"controlled calibration compiled trace gate failed: {trace_counts}")
    payload = {
        "schema": f"bayesfilter.ssl_lstm_neutra.phase8_controlled_calibration.{mode}.v1",
        "status": "PASSED_EXECUTION",
        "decision": decision,
        "mode": mode,
        "pilot_binding": {
            "path": PILOT_RECEIPT_PATH.as_posix(),
            "sha256": PILOT_RECEIPT_SHA256,
            "decision": pilot["decision"],
        },
        "smoke_binding": (
            {
                "path": SMOKE_RECEIPT_PATH.as_posix(),
                "sha256": SMOKE_RECEIPT_SHA256,
                "decision": smoke["decision"],
            }
            if smoke is not None
            else None
        ),
        "configuration": {
            "shape_per_arm": [CHAIN_COUNT, DRAW_COUNT, REPLICATION_COUNT, HORIZON],
            "block_length": BLOCK_LENGTH,
            "feature_alpha": FEATURE_ALPHA,
            "mmd_alpha": MMD_ALPHA,
            "total_alpha": TOTAL_ALPHA,
            "mean_margin": MEAN_MARGIN,
            "log_variance_margin": LOG_VARIANCE_MARGIN,
            "material_mean_anchor": 0.20,
            "material_variance_ratios": [1.25, 0.80],
            "bandwidths": list(pooled["bandwidth_candidates"]),
            "mixture_weights": [0.2] * 5,
            "mmd_tolerance_candidates": list(MMD_TOLERANCES),
            "selected_mmd_tolerance": nominated,
            "ridge_ladder": list(RIDGE_LADDER),
            "condition_number_max": CONDITION_NUMBER_MAX,
            "interval_method": "bonferroni_studentized",
            "coverage_definition": "simultaneous_20_feature_interval_coverage",
            "chain_pair_schedule": [list(row) for row in CHAIN_PAIR_SCHEDULE],
            "root_seed": list(root_seed),
            "replication_count": executed_count,
            "planned_max_replication_count": count,
        },
        "families": [FAMILIES[index].__dict__ for index in active_family_indices],
        "aggregate": aggregates,
        "sequential_stopping": {
            "enabled": mode == "nomination",
            "rule": (
                "stop only when no tolerance can meet every frozen threshold "
                "after assigning all remaining outcomes favorably"
                if mode == "nomination"
                else None
            ),
            "stopped_early": sequential_stop_reason is not None,
            "reason": sequential_stop_reason,
            "viable_tolerances_after_stop": (
                list(viable_tolerances_after_stop)
                if viable_tolerances_after_stop is not None
                else None
            ),
        },
        "compile_trace_counts": trace_counts,
        "synthetic_validity": {
            "draw_stationarity": "initial Gaussian cluster drawn from invariant N(0,R); AR recursion uses sqrt(1-phi^2)",
            "draw_mixing": "all configured absolute phi values are strictly below one",
            "marginal_standardization": "cluster/replication mixture variances sum to one",
            "arm_independence": "left and right use distinct Philox fold-in domains 11 and 22",
        },
        "validation_exact_binomial_bounds": (
            validation_bounds
        ),
        "per_replication": family_rows,
        "source_bindings": {
            "plan": {"path": PLAN_PATH.as_posix(), "sha256": _sha256(PLAN_PATH)},
            "runner": {"path": SCRIPT_PATH.as_posix(), "sha256": _sha256(SCRIPT_PATH)},
            "predictive": {"path": PREDICTIVE_SOURCE.as_posix(), "sha256": _sha256(PREDICTIVE_SOURCE)},
        },
        "run_manifest": {
            "command": shlex.join((sys.executable, *sys.argv)),
            "cwd": str(ROOT),
            "interpreter": sys.executable,
            "python_version": platform.python_version(),
            "tensorflow": tf.__version__,
            "tensorflow_probability": tfp.__version__,
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "physical_devices": [device.name for device in tf.config.list_physical_devices("GPU")],
            "logical_devices": [device.name for device in tf.config.list_logical_devices("GPU")],
            "jit_compile": True,
            "dtype": "float64",
            "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
            "git_commit": _git("rev-parse", "HEAD").strip(),
            "git_dirty": bool(_git("status", "--porcelain").strip()),
            "started_at_utc": started_at,
            "completed_at_utc": _now(),
            "wall_time_seconds": wall_time,
            "wall_cap_seconds": wall_cap_seconds,
            "output_path": output.as_posix(),
            "plan_path": PLAN_PATH.as_posix(),
        },
        "nonclaims": [
            "controlled synthetic calibration only; no G/H confirmation forecast read",
            "nomination is not validation and validation is not predictive equivalence",
            "no posterior truth, sampler ranking, model adequacy, or default claim",
            "skew and dependence families remain explanatory",
        ],
    }
    _write(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("smoke", "nomination", "validation"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-cap-seconds", type=float, required=True)
    parser.add_argument("--selected-tolerance", type=float)
    args = parser.parse_args()
    physical = tf.config.list_physical_devices("GPU")
    if not physical:
        raise CalibrationError("controlled calibration requires a visible GPU")
    for gpu in physical:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    with tf.device("/GPU:0"):
        payload = run(
            mode=args.mode,
            output=args.output,
            wall_cap_seconds=float(args.wall_cap_seconds),
            selected_tolerance=args.selected_tolerance,
        )
    print(
        "JSON_SUMMARY "
        + json.dumps(
            {
                "decision": payload["decision"],
                "selected_mmd_tolerance": payload["configuration"]["selected_mmd_tolerance"],
                "wall_time_seconds": payload["run_manifest"]["wall_time_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
