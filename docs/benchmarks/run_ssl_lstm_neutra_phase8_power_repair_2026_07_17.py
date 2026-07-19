#!/usr/bin/env python3
"""Controlled power-repair ladder for the Phase 8 predictive design."""

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
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bayesfilter.inference.predictive_equivalence as predictive  # noqa: E402


PLAN_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-neutra-phase-8-power-repair-plan-2026-07-17.md"
)
SCRIPT_PATH = Path(__file__).resolve().relative_to(ROOT)
PREDICTIVE_SOURCE = Path("bayesfilter/inference/predictive_equivalence.py")
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
FAILED_NOMINATION_PATH = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/"
    "controlled-calibration-nomination.json"
)
FAILED_NOMINATION_SHA256 = (
    "ec112880f6e9f33432ad5c12f2ccc81efd71b40a75470fca45293a7aba225b49"
)
FAILED_RUNNER_PATH = Path(
    "docs/benchmarks/run_ssl_lstm_neutra_phase8_controlled_calibration_2026_07_17.py"
)
FAILED_RUNNER_SHA256 = (
    "3378822337fe3dd079e90ac3b381ef74949e53acc937098b724ce8e767266ef7"
)
POWER_REPAIR_SMOKE_PATH = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/"
    "power-repair-smoke.json"
)
POWER_REPAIR_SMOKE_SHA256 = (
    "d0a23bcd3ebd2340c955824941ce6b726d6eee0b2c538d96a38c518188f212f3"
)
POWER_REPAIR_SMOKE_RUNNER_SHA256 = (
    "3460e49a421869bb0eea2d6fbdeaf9e914ec2d482a7ec4abf509c3daa49ee0b8"
)

CHAIN_COUNT = 4
DRAW_COUNT = 1984
REPLICATION_COUNT = 2
HORIZON = 10
BLOCK_LENGTH = 16
FEATURE_ALPHA = 0.03
MMD_ALPHA = 0.02
TOTAL_ALPHA = 0.05
MMD_TOLERANCES = (0.005, 0.01, 0.02, 0.04, 0.08, 0.16)
RIDGE_LADDER = (0.0, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6)
CONDITION_NUMBER_MAX = 1.0e8
NOMINATION_COUNT = 20
SMOKE_SEED = (15501, 15502)
NOMINATION_SEED = (16001, 16002)
CHAIN_PAIR_SCHEDULE = ((0, 1), (2, 3))
ARM_ORDER = ("B", "C", "D")


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


@dataclass(frozen=True)
class Arm:
    name: str
    mean_margin: float
    log_variance_margin: float
    equivalence_rule: Literal["symmetric_bonferroni", "iut_tost"]
    material_rule: Literal["symmetric_bonferroni"] = "symmetric_bonferroni"


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

ARMS = (
    Arm("B", 0.15, math.log(1.15), "symmetric_bonferroni"),
    Arm("C", 0.10, 0.5 * math.log(1.25), "symmetric_bonferroni"),
    Arm("D", 0.10, 0.5 * math.log(1.25), "iut_tost"),
)


class PowerRepairError(RuntimeError):
    """Raised when the controlled power repair fails closed."""


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(_absolute(path).read_bytes()).hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise PowerRepairError(f"duplicate JSON key {key!r}: {path}")
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise PowerRepairError(f"nonfinite JSON constant {value!r}: {path}")

    value = json.loads(
        _absolute(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=reject,
    )
    if not isinstance(value, dict):
        raise PowerRepairError(f"expected JSON object: {path}")
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
        raise PowerRepairError(f"refusing to overwrite receipt: {path}")
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


def _trace_count(program: Any) -> int | None:
    method = getattr(program, "experimental_get_tracing_count", None)
    return None if method is None else int(method())


def _require_gpu(*tensors: tf.Tensor, surface: str) -> None:
    devices = [str(tensor.device) for tensor in tensors]
    if not devices or any("GPU:" not in device for device in devices):
        raise PowerRepairError(f"{surface} outputs are not GPU resident: {devices}")


def _validate_bindings(*, require_power_repair_smoke: bool = False) -> dict[str, dict[str, Any]]:
    expected = (
        (PILOT_RECEIPT_PATH, PILOT_RECEIPT_SHA256),
        (SMOKE_RECEIPT_PATH, SMOKE_RECEIPT_SHA256),
        (FAILED_NOMINATION_PATH, FAILED_NOMINATION_SHA256),
        (FAILED_RUNNER_PATH, FAILED_RUNNER_SHA256),
    )
    for path, digest in expected:
        if _sha256(path) != digest:
            raise PowerRepairError(f"binding identity drift: {path}")
    pilot = _strict_json(PILOT_RECEIPT_PATH)
    smoke = _strict_json(SMOKE_RECEIPT_PATH)
    failed = _strict_json(FAILED_NOMINATION_PATH)
    if (
        pilot.get("decision")
        != "PHASE8_TARGET_PILOT_PASSED_CONTROL_CALIBRATION_REQUIRED"
        or pilot.get("split_contract", {}).get("confirmation_forecast_bank_opened")
        is not False
        or pilot.get("split_contract", {}).get("g_h_predictive_difference_computed")
        is not False
    ):
        raise PowerRepairError("target-pilot contract drift")
    if (
        smoke.get("decision")
        != "PHASE8_CONTROLLED_CALIBRATION_SMOKE_PASSED_NOMINATION_REQUIRED"
        or smoke.get("configuration", {}).get("selected_mmd_tolerance") is not None
    ):
        raise PowerRepairError("controlled smoke contract drift")
    if (
        failed.get("decision")
        != "PHASE8_CONTROLLED_NOMINATION_UNDERPOWERED_REPAIR_REQUIRED"
        or failed.get("configuration", {}).get("selected_mmd_tolerance") is not None
        or failed.get("sequential_stopping", {}).get("stopped_early") is not True
        or failed.get("sequential_stopping", {}).get("viable_tolerances_after_stop")
        != []
        or failed.get("validation_exact_binomial_bounds") is not None
    ):
        raise PowerRepairError("failed nomination contract drift")
    result = {"pilot": pilot, "smoke": smoke, "failed_nomination": failed}
    if require_power_repair_smoke:
        if _sha256(POWER_REPAIR_SMOKE_PATH) != POWER_REPAIR_SMOKE_SHA256:
            raise PowerRepairError("power-repair smoke receipt identity drift")
        repair_smoke = _strict_json(POWER_REPAIR_SMOKE_PATH)
        if (
            repair_smoke.get("status") != "PASSED_EXECUTION"
            or repair_smoke.get("decision")
            != "PHASE8_POWER_REPAIR_SMOKE_PASSED_NOMINATION_REQUIRED"
            or repair_smoke.get("mode") != "smoke"
            or repair_smoke.get("configuration", {}).get("selected_arm") is not None
            or repair_smoke.get("configuration", {}).get("selected_mmd_tolerance")
            is not None
            or repair_smoke.get("configuration", {}).get("shape_per_arm")
            != [CHAIN_COUNT, DRAW_COUNT, REPLICATION_COUNT, HORIZON]
            or [row.get("name") for row in repair_smoke.get("families", [])]
            != ["iid_null", "material_mean_persistent_pos0p20"]
            or set(repair_smoke.get("compile_trace_counts", {}).values()) != {1}
            or repair_smoke.get("source_bindings", {}).get("runner", {}).get("sha256")
            != POWER_REPAIR_SMOKE_RUNNER_SHA256
            or repair_smoke.get("run_manifest", {}).get("cuda_visible_devices") != "1"
            or repair_smoke.get("run_manifest", {}).get("jit_compile") is not True
            or repair_smoke.get("run_manifest", {}).get("trust_basis")
            != "owner_designated_managed_session_visible_gpu_trusted"
        ):
            raise PowerRepairError("power-repair smoke contract drift")
        result["power_repair_smoke"] = repair_smoke
    return result


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


@tf.function(jit_compile=True, autograph=False)
def _tost_bounds(
    estimate: tf.Tensor, standard_error: tf.Tensor, alpha: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    critical = tf.sqrt(tf.constant(2.0, tf.float64)) * tf.math.erfinv(
        2.0 * (1.0 - alpha) - 1.0
    )
    return (
        estimate - critical * standard_error,
        estimate + critical * standard_error,
        critical,
    )


def _family_paths(
    root_seed: tuple[int, int], family_index: int, replication: int
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    seed = _fold(
        _fold(tf.constant(root_seed, tf.int32), family_index), replication
    )
    family = FAMILIES[family_index]
    left = _gaussian_paths(
        _fold(seed, 11), tf.constant(family.phi, tf.float64), tf.constant(0.7, tf.float64)
    )
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
            ratios = tf.fill(
                [HORIZON], tf.constant(family.variance_ratio, tf.float64)
            )
        else:
            ratios = tf.tensor_scatter_nd_update(
                tf.ones([HORIZON], tf.float64),
                [[family.local_horizon]],
                [family.variance_ratio],
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


def _margins(arm: Arm) -> tf.Tensor:
    return tf.concat(
        (
            tf.fill([HORIZON], tf.constant(arm.mean_margin, tf.float64)),
            tf.fill(
                [HORIZON], tf.constant(arm.log_variance_margin, tf.float64)
            ),
        ),
        axis=0,
    )


def _feature_status(
    bonferroni_lower: tf.Tensor,
    bonferroni_upper: tf.Tensor,
    tost_lower: tf.Tensor,
    tost_upper: tf.Tensor,
    margins: tf.Tensor,
    *,
    equivalence_rule: str,
) -> str:
    material = bool(
        tf.reduce_any(
            tf.logical_or(
                bonferroni_lower > margins, bonferroni_upper < -margins
            )
        )
    )
    if equivalence_rule == "symmetric_bonferroni":
        equivalent = bool(
            tf.reduce_all(
                tf.logical_and(
                    bonferroni_lower > -margins, bonferroni_upper < margins
                )
            )
        )
    elif equivalence_rule == "iut_tost":
        equivalent = bool(
            tf.reduce_all(
                tf.logical_and(tost_lower > -margins, tost_upper < margins)
            )
        )
    else:
        raise PowerRepairError("unknown equivalence rule")
    if material:
        return "MATERIAL_DIFFERENCE"
    if equivalent:
        return "PASS"
    return "INCONCLUSIVE_UNDERPOWERED"


def _mmd_status(lower: tf.Tensor, upper: tf.Tensor, tolerance: float) -> str:
    bound = tf.constant(tolerance, tf.float64)
    if bool(lower > bound):
        return "MATERIAL_DIFFERENCE"
    if bool(upper < bound):
        return "PASS"
    return "INCONCLUSIVE_UNDERPOWERED"


def _combine_status(feature_status: str, mmd_status: str) -> str:
    if "MATERIAL_DIFFERENCE" in (feature_status, mmd_status):
        return "MATERIAL_DIFFERENCE"
    if feature_status == "PASS" and mmd_status == "PASS":
        return "PASS"
    return "INCONCLUSIVE_UNDERPOWERED"


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
        (
            2.0 * left_features.influence_values,
            -2.0 * right_features.influence_values,
        ),
        axis=0,
    )
    covariance = predictive.chain_batch_long_run_covariance(
        influences,
        block_length=BLOCK_LENGTH,
        ridge_ladder=RIDGE_LADDER,
        condition_number_max=CONDITION_NUMBER_MAX,
    )
    if not covariance.inference_admissible:
        raise PowerRepairError("power-repair feature covariance is inadmissible")
    standard_error = tf.sqrt(
        tf.linalg.diag_part(covariance.regularized_covariance)
    )
    bonferroni = predictive.simultaneous_feature_intervals(
        estimate,
        feature_alpha=FEATURE_ALPHA,
        method="bonferroni_studentized",
        standard_error=standard_error,
    )
    tost_lower, tost_upper, tost_critical = _tost_bounds(
        estimate, standard_error, tf.constant(FEATURE_ALPHA, tf.float64)
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
        raise PowerRepairError("power-repair MMD interval is inadmissible")
    _require_gpu(
        estimate,
        covariance.regularized_covariance,
        bonferroni.lower,
        bonferroni.upper,
        tost_lower,
        tost_upper,
        mmd.kernel_contrast_sequence,
        mmd_interval.lower,
        mmd_interval.upper,
        surface="controlled power repair",
    )
    coverage = bool(
        tf.reduce_all(
            tf.logical_and(bonferroni.lower <= truth, truth <= bonferroni.upper)
        )
    )
    status_by_arm: dict[str, dict[str, str]] = {}
    primary_by_arm: dict[str, str] = {}
    for arm in ARMS:
        feature_status = _feature_status(
            bonferroni.lower,
            bonferroni.upper,
            tost_lower,
            tost_upper,
            _margins(arm),
            equivalence_rule=arm.equivalence_rule,
        )
        primary_by_arm[arm.name] = feature_status
        status_by_arm[arm.name] = {
            str(tolerance): _combine_status(
                feature_status,
                _mmd_status(mmd_interval.lower, mmd_interval.upper, tolerance),
            )
            for tolerance in MMD_TOLERANCES
        }
    return {
        "coverage": coverage,
        "primary_status_by_arm": primary_by_arm,
        "status_by_arm_and_tolerance": status_by_arm,
        "feature_estimate_max_abs": float(tf.reduce_max(tf.abs(estimate))),
        "bonferroni_interval_max_width": float(
            tf.reduce_max(bonferroni.upper - bonferroni.lower)
        ),
        "tost_interval_max_width": float(tf.reduce_max(tost_upper - tost_lower)),
        "bonferroni_critical": float(bonferroni.critical_value),
        "tost_critical": float(tost_critical),
        "mmd_estimate": float(mmd.squared_mmd_linear),
        "mmd_lower": float(mmd_interval.lower),
        "mmd_upper": float(mmd_interval.upper),
        "ridge_multiplier": float(covariance.selected_ridge_multiplier),
        "condition_number": float(covariance.condition_number),
    }


def _aggregate(rows: list[dict[str, Any]], arm_name: str) -> dict[str, Any]:
    by_tolerance: dict[str, dict[str, int]] = {}
    for tolerance in MMD_TOLERANCES:
        key = str(tolerance)
        statuses = [row["status_by_arm_and_tolerance"][arm_name][key] for row in rows]
        by_tolerance[key] = {
            "pass_count": statuses.count("PASS"),
            "material_difference_count": statuses.count("MATERIAL_DIFFERENCE"),
            "inconclusive_count": statuses.count("INCONCLUSIVE_UNDERPOWERED"),
        }
    primary = [row["primary_status_by_arm"][arm_name] for row in rows]
    return {
        "replication_count": len(rows),
        "coverage_count": sum(row["coverage"] for row in rows),
        "primary_pass_count": primary.count("PASS"),
        "primary_material_difference_count": primary.count("MATERIAL_DIFFERENCE"),
        "by_tolerance": by_tolerance,
        "max_bonferroni_interval_width": max(
            row["bonferroni_interval_max_width"] for row in rows
        ),
        "max_tost_interval_width": max(
            row["tost_interval_max_width"] for row in rows
        ),
        "max_condition_number": max(row["condition_number"] for row in rows),
    }


def _aggregates(
    family_rows: dict[str, list[dict[str, Any]]]
) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        arm.name: {
            name: _aggregate(rows, arm.name) for name, rows in family_rows.items()
        }
        for arm in ARMS
    }


def _candidate_pass(
    aggregates: dict[str, dict[str, dict[str, Any]]],
    arm_name: str,
    tolerance: float,
) -> bool:
    key = str(tolerance)
    for family in FAMILIES:
        if family.role == "explanatory":
            continue
        row = aggregates[arm_name][family.name]
        if row["replication_count"] != NOMINATION_COUNT or row["coverage_count"] < 18:
            return False
        counts = row["by_tolerance"][key]
        if family.role == "equivalence":
            if counts["pass_count"] < 16 or counts["material_difference_count"] > 1:
                return False
        elif counts["pass_count"] > 1 or counts["material_difference_count"] < 16:
            return False
    return True


def _viable_candidates(
    aggregates: dict[str, dict[str, dict[str, Any]]], completed_count: int
) -> tuple[tuple[str, float], ...]:
    if not 0 <= completed_count <= NOMINATION_COUNT:
        raise PowerRepairError("invalid repair nomination replication count")
    remaining = NOMINATION_COUNT - completed_count
    viable: list[tuple[str, float]] = []
    for arm_name in ARM_ORDER:
        for tolerance in MMD_TOLERANCES:
            key = str(tolerance)
            candidate_viable = True
            for family in FAMILIES:
                if family.role == "explanatory":
                    continue
                row = aggregates[arm_name][family.name]
                counts = row["by_tolerance"][key]
                if row["coverage_count"] + remaining < 18:
                    candidate_viable = False
                elif family.role == "equivalence" and (
                    counts["pass_count"] + remaining < 16
                    or counts["material_difference_count"] > 1
                ):
                    candidate_viable = False
                elif family.role == "material" and (
                    counts["material_difference_count"] + remaining < 16
                    or counts["pass_count"] > 1
                ):
                    candidate_viable = False
                if not candidate_viable:
                    break
            if candidate_viable:
                viable.append((arm_name, tolerance))
    return tuple(viable)


def run(*, mode: str, output: Path, wall_cap_seconds: float) -> dict[str, Any]:
    if mode not in {"smoke", "nomination"}:
        raise PowerRepairError("validation and HMC acquisition remain closed")
    if not math.isfinite(wall_cap_seconds) or wall_cap_seconds <= 0.0:
        raise PowerRepairError("wall cap must be positive and finite")
    bindings = _validate_bindings(require_power_repair_smoke=mode == "nomination")
    bandwidths = tf.constant(
        bindings["pilot"]["pooled_calibration"]["bandwidth_candidates"],
        tf.float64,
    )
    if tuple(bandwidths.shape) != (5,):
        raise PowerRepairError("pilot bandwidth shape drift")
    count = 1 if mode == "smoke" else NOMINATION_COUNT
    root_seed = SMOKE_SEED if mode == "smoke" else NOMINATION_SEED
    family_indices = (0, 4) if mode == "smoke" else tuple(range(len(FAMILIES)))
    started_at = _now()
    started = time.perf_counter()
    family_rows: dict[str, list[dict[str, Any]]] = {
        FAMILIES[index].name: [] for index in family_indices
    }
    stop_reason = None
    viable_after_stop: tuple[tuple[str, float], ...] | None = None
    for replication in range(count):
        for family_index in family_indices:
            family = FAMILIES[family_index]
            left, right, truth = _family_paths(root_seed, family_index, replication)
            _require_gpu(left, right, truth, surface="power-repair family generation")
            family_rows[family.name].append(
                _one_evidence(left, right, truth, bandwidths=bandwidths)
            )
        if time.perf_counter() - started > wall_cap_seconds:
            raise PowerRepairError("power-repair wall cap exceeded")
        if mode == "nomination":
            partial = _aggregates(family_rows)
            viable = _viable_candidates(partial, replication + 1)
            print(
                "POWER_REPAIR_PROGRESS "
                + json.dumps(
                    {
                        "completed_replications": replication + 1,
                        "elapsed_seconds": time.perf_counter() - started,
                        "viable_candidate_count": len(viable),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if replication + 1 < count and not viable:
                stop_reason = (
                    "all arm/tolerance candidates are unable to meet the frozen "
                    "20-replication thresholds"
                )
                viable_after_stop = viable
                break
    aggregates = _aggregates(family_rows)
    executed_count = len(next(iter(family_rows.values())))
    selected_arm = None
    selected_tolerance = None
    if mode == "smoke":
        decision = "PHASE8_POWER_REPAIR_SMOKE_PASSED_NOMINATION_REQUIRED"
    else:
        for arm_name in ARM_ORDER:
            selected_tolerance = next(
                (
                    tolerance
                    for tolerance in MMD_TOLERANCES
                    if _candidate_pass(aggregates, arm_name, tolerance)
                ),
                None,
            )
            if selected_tolerance is not None:
                selected_arm = arm_name
                break
        decision = (
            "PHASE8_POWER_REPAIR_NOMINATION_PASSED_FRESH_VALIDATION_REQUIRED"
            if selected_arm is not None
            else "PHASE8_POWER_REPAIR_NOMINATION_UNDERPOWERED_STOP"
        )
    trace_counts = {
        "gaussian_paths": _trace_count(_gaussian_paths),
        "mean_log_variance_influence": _trace_count(
            predictive._mean_log_variance_influence_xla
        ),
        "long_run_covariance": _trace_count(predictive._long_run_covariance_xla),
        "bonferroni_interval": _trace_count(predictive._bonferroni_interval_xla),
        "tost_bounds": _trace_count(_tost_bounds),
        "cluster_kernel": _trace_count(predictive._cluster_kernel_xla),
        "mmd_block_interval": _trace_count(predictive._mmd_block_interval_xla),
    }
    if any(value != 1 for value in trace_counts.values()):
        raise PowerRepairError(f"power-repair compiled trace gate failed: {trace_counts}")
    wall_time = time.perf_counter() - started
    payload = {
        "schema": f"bayesfilter.ssl_lstm_neutra.phase8_power_repair.{mode}.v1",
        "status": "PASSED_EXECUTION",
        "decision": decision,
        "mode": mode,
        "bindings": {
            "target_pilot": {
                "path": PILOT_RECEIPT_PATH.as_posix(),
                "sha256": PILOT_RECEIPT_SHA256,
                "decision": bindings["pilot"]["decision"],
            },
            "controlled_smoke": {
                "path": SMOKE_RECEIPT_PATH.as_posix(),
                "sha256": SMOKE_RECEIPT_SHA256,
                "decision": bindings["smoke"]["decision"],
            },
            "failed_nomination": {
                "path": FAILED_NOMINATION_PATH.as_posix(),
                "sha256": FAILED_NOMINATION_SHA256,
                "decision": bindings["failed_nomination"]["decision"],
            },
            "failed_runner": {
                "path": FAILED_RUNNER_PATH.as_posix(),
                "sha256": FAILED_RUNNER_SHA256,
            },
            "power_repair_smoke": (
                {
                    "path": POWER_REPAIR_SMOKE_PATH.as_posix(),
                    "sha256": POWER_REPAIR_SMOKE_SHA256,
                    "decision": bindings["power_repair_smoke"]["decision"],
                }
                if mode == "nomination"
                else None
            ),
        },
        "configuration": {
            "shape_per_arm": [CHAIN_COUNT, DRAW_COUNT, REPLICATION_COUNT, HORIZON],
            "block_length": BLOCK_LENGTH,
            "feature_alpha": FEATURE_ALPHA,
            "mmd_alpha": MMD_ALPHA,
            "total_alpha": TOTAL_ALPHA,
            "arms": [asdict(arm) for arm in ARMS],
            "material_mean_anchor": 0.20,
            "material_variance_ratios": [1.25, 0.80],
            "bandwidths": list(bandwidths.numpy()),
            "mixture_weights": [0.2] * 5,
            "mmd_tolerance_candidates": list(MMD_TOLERANCES),
            "selected_arm": selected_arm,
            "selected_mmd_tolerance": selected_tolerance,
            "selection_order": list(ARM_ORDER),
            "ridge_ladder": list(RIDGE_LADDER),
            "condition_number_max": CONDITION_NUMBER_MAX,
            "coverage_interval": "bonferroni_studentized_20_feature",
            "material_interval": "bonferroni_studentized_20_feature",
            "arm_d_equivalence_interval": "componentwise_one_sided_alpha_iut_tost",
            "root_seed": list(root_seed),
            "replication_count": executed_count,
            "planned_max_replication_count": count,
        },
        "families": [asdict(FAMILIES[index]) for index in family_indices],
        "aggregate_by_arm": aggregates,
        "sequential_stopping": {
            "enabled": mode == "nomination",
            "rule": (
                "stop only when no arm/tolerance can meet every frozen threshold "
                "after assigning all remaining outcomes favorably"
                if mode == "nomination"
                else None
            ),
            "stopped_early": stop_reason is not None,
            "reason": stop_reason,
            "viable_candidates_after_stop": (
                [list(row) for row in viable_after_stop]
                if viable_after_stop is not None
                else None
            ),
        },
        "compile_trace_counts": trace_counts,
        "synthetic_validity": {
            "draw_stationarity": (
                "initial Gaussian cluster drawn from invariant N(0,R); "
                "AR recursion uses sqrt(1-phi^2)"
            ),
            "draw_mixing": "all configured absolute phi values are strictly below one",
            "marginal_standardization": (
                "cluster/replication mixture variances sum to one"
            ),
            "arm_independence": (
                "left and right use distinct Philox fold-in domains 11 and 22"
            ),
            "candidate_fairness": (
                "B/C/D reuse the same generated paths, estimates, covariance, "
                "Bonferroni interval, and MMD interval within each family/replication"
            ),
        },
        "per_replication": family_rows,
        "source_bindings": {
            "plan": {"path": PLAN_PATH.as_posix(), "sha256": _sha256(PLAN_PATH)},
            "runner": {"path": SCRIPT_PATH.as_posix(), "sha256": _sha256(SCRIPT_PATH)},
            "predictive": {
                "path": PREDICTIVE_SOURCE.as_posix(),
                "sha256": _sha256(PREDICTIVE_SOURCE),
            },
        },
        "run_manifest": {
            "command": shlex.join((sys.executable, *sys.argv)),
            "cwd": str(ROOT),
            "interpreter": sys.executable,
            "python_version": platform.python_version(),
            "tensorflow": tf.__version__,
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
            "controlled synthetic power repair only; no G/H confirmation forecast read",
            "repair nomination is not fresh validation or predictive equivalence",
            "no posterior truth, sampler ranking, model adequacy, or default claim",
            "1984 draws are a feasibility arm and do not authorize HMC acquisition",
            "skew and dependence families remain explanatory",
        ],
    }
    _write(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "nomination"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-cap-seconds", type=float, required=True)
    args = parser.parse_args()
    physical = tf.config.list_physical_devices("GPU")
    if not physical:
        raise PowerRepairError("controlled power repair requires a visible GPU")
    for gpu in physical:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    with tf.device("/GPU:0"):
        payload = run(
            mode=args.mode,
            output=args.output,
            wall_cap_seconds=float(args.wall_cap_seconds),
        )
    print(
        "JSON_SUMMARY "
        + json.dumps(
            {
                "decision": payload["decision"],
                "selected_arm": payload["configuration"]["selected_arm"],
                "selected_mmd_tolerance": payload["configuration"]["selected_mmd_tolerance"],
                "wall_time_seconds": payload["run_manifest"]["wall_time_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
