#!/usr/bin/env python3
"""Feasibility-only sample-size and margin preflight for Phase 8."""

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
import tensorflow_probability as tfp


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bayesfilter.inference.predictive_equivalence as predictive  # noqa: E402


PLAN_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-neutra-phase-8-sample-size-margin-"
    "preflight-plan-2026-07-17.md"
)
SCRIPT_PATH = Path(__file__).resolve().relative_to(ROOT)
PREDICTIVE_SOURCE = Path("bayesfilter/inference/predictive_equivalence.py")
TARGET_PILOT_PATH = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/"
    "target-pilot-repair-03.json"
)
TARGET_PILOT_SHA256 = (
    "5ae511c248e222edf14660c91c4a48412706c6f452298ebe5e144bbe8f01c098"
)
FAILED_448_PATH = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/"
    "controlled-calibration-nomination.json"
)
FAILED_448_SHA256 = (
    "ec112880f6e9f33432ad5c12f2ccc81efd71b40a75470fca45293a7aba225b49"
)
FAILED_1984_PATH = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/"
    "power-repair-nomination.json"
)
FAILED_1984_SHA256 = (
    "56a34c4a254c38d89f682a22c4100d7df56d9aef460ae06d81e45de9d684e729"
)
PHASE7_RECEIPT_PATH = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/"
    "phase-7-retained-admission/retained-acquisition.json"
)
PHASE7_RECEIPT_SHA256 = (
    "b79e5f6041e284de40bbd3834cc909fd12f45d012f172e570acccaa62dbe31a5"
)
SMOKE_RECEIPT_PATH = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/"
    "sample-size-margin-preflight-smoke.json"
)
SMOKE_RECEIPT_SHA256 = (
    "7eaf2b17c56cad4b523f981e0016e6b10366573d160efae92be88cdb3af4c224"
)

CHAIN_COUNT = 4
MAX_DRAW_COUNT = 8192
DRAW_GRID = (1984, 3072, 4096, 6144, 8192)
FORECAST_REPLICATION_COUNT = 2
HORIZON = 10
FEATURE_COUNT = 20
BLOCK_LENGTH = 16
FEATURE_ALPHA = 0.03
MMD_ALPHA = 0.02
MMD_TOLERANCES = (0.005, 0.01, 0.02, 0.04, 0.08, 0.16)
RIDGE_LADDER = (0.0, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6)
CONDITION_NUMBER_MAX = 1.0e8
PILOT_REPLICATIONS_MATERIAL = 4
PILOT_REPLICATIONS_SMOKE = 1
MONTE_CARLO_COUNT_MATERIAL = 20_000
MONTE_CARLO_COUNT_SMOKE = 512
PILOT_SEED = (17001, 17002)
MONTE_CARLO_SEED = (18001, 18002)
CHAIN_PAIR_SCHEDULE = ((0, 1), (2, 3))
MONTE_CARLO_BATCH_SIZE = 5000
MINIMUM_COVERAGE = 0.90
MINIMUM_REQUIRED_DECISION = 0.80
MAXIMUM_FALSE_DECISION = 0.05


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
class Scenario:
    name: str
    mean_margin: float
    log_variance_margin: float
    equivalence_rule: Literal["symmetric_bonferroni", "iut_tost"]
    role: Literal["historical_contract", "arithmetic_sensitivity"]


FAMILIES = (
    Family("iid_null", "equivalence", phi=0.0),
    Family("ar_null", "equivalence"),
    Family("trueeq_mean_persistent_0p05", "equivalence", mean_shift=0.05),
    Family(
        "trueeq_variance_persistent_1p05", "equivalence", variance_ratio=1.05
    ),
    Family("material_mean_persistent_pos0p20", "material", mean_shift=0.20),
    Family("material_mean_persistent_neg0p20", "material", mean_shift=-0.20),
    Family(
        "material_mean_local_h1_pos0p20",
        "material",
        mean_shift=0.20,
        local_horizon=0,
    ),
    Family(
        "material_mean_local_h1_neg0p20",
        "material",
        mean_shift=-0.20,
        local_horizon=0,
    ),
    Family(
        "material_variance_persistent_1p25", "material", variance_ratio=1.25
    ),
    Family(
        "material_variance_persistent_0p80", "material", variance_ratio=0.80
    ),
    Family(
        "material_variance_local_h1_1p25",
        "material",
        variance_ratio=1.25,
        local_horizon=0,
    ),
    Family("skew_explanatory", "explanatory", skew=0.35),
    Family("dependence_explanatory", "explanatory", horizon_rho=0.9),
)

SCENARIOS = (
    Scenario(
        "historical_original_symmetric",
        0.15,
        math.log(1.15),
        "symmetric_bonferroni",
        "historical_contract",
    ),
    Scenario(
        "historical_repair_tost",
        0.10,
        0.5 * math.log(1.25),
        "iut_tost",
        "historical_contract",
    ),
    Scenario(
        "anchor_midpoint_tost",
        0.5 * (0.05 + 0.20),
        0.5 * (math.log(1.05) + math.log(1.25)),
        "iut_tost",
        "arithmetic_sensitivity",
    ),
)


class PreflightError(RuntimeError):
    """Raised when the sample-size preflight fails closed."""


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(_absolute(path).read_bytes()).hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise PreflightError(f"duplicate JSON key {key!r}: {path}")
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise PreflightError(f"nonfinite JSON constant {value!r}: {path}")

    payload = json.loads(
        _absolute(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=reject,
    )
    if not isinstance(payload, dict):
        raise PreflightError(f"expected JSON object: {path}")
    return payload


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
        raise PreflightError(f"refusing to overwrite receipt: {path}")
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
        raise PreflightError(f"{surface} outputs are not GPU resident: {devices}")


def _validate_bindings(*, require_smoke: bool = False) -> dict[str, dict[str, Any]]:
    expected = (
        (TARGET_PILOT_PATH, TARGET_PILOT_SHA256),
        (FAILED_448_PATH, FAILED_448_SHA256),
        (FAILED_1984_PATH, FAILED_1984_SHA256),
        (PHASE7_RECEIPT_PATH, PHASE7_RECEIPT_SHA256),
    )
    for path, digest in expected:
        if _sha256(path) != digest:
            raise PreflightError(f"binding identity drift: {path}")
    pilot = _strict_json(TARGET_PILOT_PATH)
    failed_448 = _strict_json(FAILED_448_PATH)
    failed_1984 = _strict_json(FAILED_1984_PATH)
    phase7 = _strict_json(PHASE7_RECEIPT_PATH)
    split = pilot.get("split_contract", {})
    if (
        pilot.get("decision")
        != "PHASE8_TARGET_PILOT_PASSED_CONTROL_CALIBRATION_REQUIRED"
        or split.get("confirmation_forecast_bank_opened") is not False
        or split.get("g_h_predictive_difference_computed") is not False
    ):
        raise PreflightError("target-pilot no-confirmation contract drift")
    if (
        failed_448.get("decision")
        != "PHASE8_CONTROLLED_NOMINATION_UNDERPOWERED_REPAIR_REQUIRED"
        or failed_448.get("configuration", {}).get("selected_mmd_tolerance")
        is not None
    ):
        raise PreflightError("448-draw failed-baseline contract drift")
    if (
        failed_1984.get("decision")
        != "PHASE8_POWER_REPAIR_NOMINATION_UNDERPOWERED_STOP"
        or failed_1984.get("configuration", {}).get("selected_arm") is not None
        or failed_1984.get("configuration", {}).get("selected_mmd_tolerance")
        is not None
    ):
        raise PreflightError("1984-draw failed-baseline contract drift")
    if phase7.get("decision") != "PHASE7_RETAINED_ADMISSION_PASSED_PHASE8_HANDOFF":
        raise PreflightError("Phase 7 timing receipt contract drift")
    result = {
        "pilot": pilot,
        "failed_448": failed_448,
        "failed_1984": failed_1984,
        "phase7": phase7,
    }
    if require_smoke:
        if _sha256(SMOKE_RECEIPT_PATH) != SMOKE_RECEIPT_SHA256:
            raise PreflightError("sample-size preflight smoke identity drift")
        smoke = _strict_json(SMOKE_RECEIPT_PATH)
        if (
            smoke.get("decision")
            != "PHASE8_SAMPLE_SIZE_PREFLIGHT_SMOKE_PASSED_MATERIAL_REQUIRED"
            or smoke.get("configuration", {}).get("margin_selection") is not None
            or smoke.get("configuration", {}).get("mmd_tolerance_selection") is not None
        ):
            raise PreflightError("sample-size preflight smoke contract drift")
        result["smoke"] = smoke
    return result


def _fold(seed: tf.Tensor, value: int) -> tf.Tensor:
    return tf.random.experimental.stateless_fold_in(
        seed, tf.constant(value, tf.int32), alg="philox"
    )


@tf.function(jit_compile=True, autograph=False)
def _gaussian_paths(seed: tf.Tensor, phi: tf.Tensor, rho: tf.Tensor) -> tf.Tensor:
    index = tf.range(HORIZON, dtype=tf.float64)
    horizon_covariance = tf.pow(rho, tf.abs(index[:, None] - index[None, :]))
    horizon_factor = tf.linalg.cholesky(horizon_covariance)
    draw_noise = tf.random.stateless_normal(
        [CHAIN_COUNT, MAX_DRAW_COUNT, HORIZON],
        _fold(seed, 1),
        dtype=tf.float64,
        alg="philox",
    ) @ tf.transpose(horizon_factor)
    initial = tf.random.stateless_normal(
        [CHAIN_COUNT, HORIZON],
        _fold(seed, 2),
        dtype=tf.float64,
        alg="philox",
    ) @ tf.transpose(horizon_factor)
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
        [CHAIN_COUNT, MAX_DRAW_COUNT, FORECAST_REPLICATION_COUNT, HORIZON],
        _fold(seed, 3),
        dtype=tf.float64,
        alg="philox",
    ) @ tf.transpose(horizon_factor)
    return tf.sqrt(tf.constant(0.65, tf.float64)) * clusters[:, :, None, :] + tf.sqrt(
        tf.constant(0.35, tf.float64)
    ) * replication_noise


@tf.function(jit_compile=True, autograph=False)
def _joint_normal_draws(
    seed: tf.Tensor, factor: tf.Tensor, count: tf.Tensor
) -> tf.Tensor:
    noise = tf.random.stateless_normal(
        [MONTE_CARLO_BATCH_SIZE, FEATURE_COUNT],
        seed,
        dtype=tf.float64,
        alg="philox",
    )
    return (noise @ tf.transpose(factor))[:count]


def _family_paths(
    family: Family, family_index: int, replication: int
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    seed = _fold(
        _fold(tf.constant(PILOT_SEED, tf.int32), family_index), replication
    )
    left = _gaussian_paths(
        _fold(seed, 11),
        tf.constant(family.phi, tf.float64),
        tf.constant(0.7, tf.float64),
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
    return left, right, tf.concat((-shift, -log_ratio), axis=0)


def _margins(scenario: Scenario) -> tf.Tensor:
    return tf.concat(
        (
            tf.fill([HORIZON], tf.constant(scenario.mean_margin, tf.float64)),
            tf.fill(
                [HORIZON],
                tf.constant(scenario.log_variance_margin, tf.float64),
            ),
        ),
        axis=0,
    )


def _feature_status_masks(
    estimates: tf.Tensor,
    standard_error: tf.Tensor,
    scenario: Scenario,
) -> dict[str, tf.Tensor]:
    margins = _margins(scenario)
    bonferroni_critical = tfp.distributions.Normal(
        tf.constant(0.0, tf.float64), tf.constant(1.0, tf.float64)
    ).quantile(1.0 - FEATURE_ALPHA / (2.0 * FEATURE_COUNT))
    tost_critical = tfp.distributions.Normal(
        tf.constant(0.0, tf.float64), tf.constant(1.0, tf.float64)
    ).quantile(1.0 - FEATURE_ALPHA)
    bonferroni_half = bonferroni_critical * standard_error
    lower = estimates - bonferroni_half
    upper = estimates + bonferroni_half
    material = tf.reduce_any(
        tf.logical_or(lower > margins, upper < -margins), axis=1
    )
    if scenario.equivalence_rule == "symmetric_bonferroni":
        equivalent = tf.reduce_all(
            tf.logical_and(lower > -margins, upper < margins), axis=1
        )
    elif scenario.equivalence_rule == "iut_tost":
        tost_lower = estimates - tost_critical * standard_error
        tost_upper = estimates + tost_critical * standard_error
        equivalent = tf.reduce_all(
            tf.logical_and(tost_lower > -margins, tost_upper < margins), axis=1
        )
    else:
        raise PreflightError("unknown equivalence rule")
    equivalent = tf.logical_and(equivalent, tf.logical_not(material))
    inconclusive = tf.logical_not(tf.logical_or(equivalent, material))
    return {
        "pass": equivalent,
        "material": material,
        "inconclusive": inconclusive,
    }


def _frechet_bounds(left_probability: float, right_probability: float) -> tuple[float, float]:
    if not 0.0 <= left_probability <= 1.0 or not 0.0 <= right_probability <= 1.0:
        raise PreflightError("Frechet probabilities must lie in [0,1]")
    return (
        max(0.0, left_probability + right_probability - 1.0),
        min(left_probability, right_probability),
    )


def _wilson_interval(successes: int, count: int, confidence: float = 0.95) -> tuple[float, float]:
    if not 0 <= successes <= count or count < 1:
        raise PreflightError("invalid Wilson count")
    z = float(
        tfp.distributions.Normal(
            tf.constant(0.0, tf.float64), tf.constant(1.0, tf.float64)
        ).quantile(0.5 + confidence / 2.0)
    )
    proportion = successes / count
    denominator = 1.0 + z * z / count
    center = (proportion + z * z / (2.0 * count)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / count + z * z / (4.0 * count * count))
        / denominator
    )
    return center - radius, center + radius


def _ceil_block(value: float) -> int | None:
    if not math.isfinite(value) or value <= 0.0:
        return None
    return int(BLOCK_LENGTH * math.ceil(value / BLOCK_LENGTH))


def _required_draws(
    *,
    standard_error_1984: float,
    clearance: float,
    critical: float,
    power: float,
) -> int | None:
    if standard_error_1984 <= 0.0 or clearance <= 0.0:
        return None
    z_power = float(tfp.distributions.Normal(0.0, 1.0).quantile(power))
    required = 1984.0 * (
        standard_error_1984 * (critical + z_power) / clearance
    ) ** 2
    return _ceil_block(required)


def _observed_standard_errors(failed_1984: dict[str, Any]) -> dict[str, float]:
    normal = tfp.distributions.Normal(
        tf.constant(0.0, tf.float64), tf.constant(1.0, tf.float64)
    )
    critical_bonferroni = float(
        normal.quantile(1.0 - FEATURE_ALPHA / (2.0 * FEATURE_COUNT))
    )
    critical_tost = float(normal.quantile(1.0 - FEATURE_ALPHA))
    rows = failed_1984["per_replication"]
    result: dict[str, float] = {}
    for family_name, family_rows in rows.items():
        bonferroni = max(
            row["bonferroni_interval_max_width"] for row in family_rows
        ) / (2.0 * critical_bonferroni)
        tost = max(row["tost_interval_max_width"] for row in family_rows) / (
            2.0 * critical_tost
        )
        if not math.isclose(bonferroni, tost, rel_tol=2.0e-10, abs_tol=1.0e-14):
            raise PreflightError(f"historical interval-width mismatch: {family_name}")
        result[family_name] = max(bonferroni, tost)
    return result


def _analytical_preflight(failed_1984: dict[str, Any]) -> dict[str, Any]:
    standard_errors = _observed_standard_errors(failed_1984)
    normal = tfp.distributions.Normal(
        tf.constant(0.0, tf.float64), tf.constant(1.0, tf.float64)
    )
    critical_bonferroni = float(
        normal.quantile(1.0 - FEATURE_ALPHA / (2.0 * FEATURE_COUNT))
    )
    critical_tost = float(normal.quantile(1.0 - FEATURE_ALPHA))
    requirements = (
        (
            "trueeq_mean_0p05_historical_repair",
            "trueeq_mean_persistent_0p05",
            0.10 - 0.05,
            critical_tost,
        ),
        (
            "trueeq_variance_1p05_historical_repair",
            "trueeq_variance_persistent_1p05",
            0.5 * math.log(1.25) - math.log(1.05),
            critical_tost,
        ),
        (
            "material_mean_0p20_historical_repair",
            "material_mean_persistent_pos0p20",
            0.20 - 0.10,
            critical_bonferroni,
        ),
        (
            "material_variance_1p25_historical_repair",
            "material_variance_persistent_1p25",
            math.log(1.25) - 0.5 * math.log(1.25),
            critical_bonferroni,
        ),
        (
            "material_mean_0p20_historical_original",
            "material_mean_persistent_pos0p20",
            0.20 - 0.15,
            critical_bonferroni,
        ),
    )
    rows = []
    for name, family_name, clearance, critical in requirements:
        se = standard_errors[family_name]
        rows.append(
            {
                "name": name,
                "family": family_name,
                "worst_observed_standard_error_at_1984": se,
                "clearance": clearance,
                "critical_value": critical,
                "required_draws_80pct_single_limiting_coordinate": _required_draws(
                    standard_error_1984=se,
                    clearance=clearance,
                    critical=critical,
                    power=0.80,
                ),
                "required_draws_90pct_single_limiting_coordinate": _required_draws(
                    standard_error_1984=se,
                    clearance=clearance,
                    critical=critical,
                    power=0.90,
                ),
                "required_draws_80pct_conservative_20_coordinate_lower_bound": _required_draws(
                    standard_error_1984=se,
                    clearance=clearance,
                    critical=critical,
                    power=1.0 - (1.0 - 0.80) / FEATURE_COUNT,
                ),
                "required_draws_90pct_conservative_20_coordinate_lower_bound": _required_draws(
                    standard_error_1984=se,
                    clearance=clearance,
                    critical=critical,
                    power=1.0 - (1.0 - 0.90) / FEATURE_COUNT,
                ),
            }
        )
    return {
        "scaling_formula": "SE_N = SE_1984 * sqrt(1984/N)",
        "draw_rounding": "upward_to_multiple_of_16",
        "single_coordinate_formula": (
            "N = 1984 * [SE_1984 * (critical + z_power) / clearance]^2"
        ),
        "joint_lower_bound": (
            "replace power by 1-(1-family_power)/20; conservative Bonferroni "
            "lower bound, not an exact joint-power calculation"
        ),
        "bonferroni_critical": critical_bonferroni,
        "tost_critical": critical_tost,
        "requirements": rows,
    }


def _pilot_one(
    left: tf.Tensor,
    right: tf.Tensor,
    truth: tf.Tensor,
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
        raise PreflightError("preflight feature covariance is inadmissible")
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
        raise PreflightError("preflight MMD interval is inadmissible")
    _require_gpu(
        estimate,
        covariance.regularized_covariance,
        mmd.kernel_contrast_sequence,
        mmd_interval.standard_error,
        surface="sample-size preflight pilot",
    )
    mmd_critical = float(mmd_interval.critical_value)
    batches = predictive.chain_batch_means(
        mmd.kernel_contrast_sequence, block_length=BLOCK_LENGTH
    )
    anchored = batches - batches[:, :1]
    centered = anchored - tf.reduce_mean(anchored, axis=1, keepdims=True)
    batch_count = batches.shape[1]
    pair_count = batches.shape[0]
    pair_variances = tf.reduce_sum(tf.square(centered), axis=1) / tf.cast(
        batch_count - 1, tf.float64
    )
    pair_mean_variance_terms = pair_variances / tf.cast(
        pair_count * pair_count * batch_count, tf.float64
    )
    variance = tf.reduce_sum(pair_mean_variance_terms)
    degrees_of_freedom = float(
        tf.square(variance)
        / tf.reduce_sum(
            tf.square(pair_mean_variance_terms)
            / tf.cast(batch_count - 1, tf.float64)
        )
    )
    return {
        "truth": list(truth.numpy()),
        "estimate": list(estimate.numpy()),
        "covariance_at_8192": covariance.regularized_covariance.numpy().tolist(),
        "condition_number": float(covariance.condition_number),
        "ridge_multiplier": float(covariance.selected_ridge_multiplier),
        "mmd_estimate": float(mmd.squared_mmd_linear),
        "mmd_standard_error_at_8192": float(mmd_interval.standard_error),
        "mmd_critical_at_8192": mmd_critical,
        "mmd_degrees_of_freedom_at_8192": degrees_of_freedom,
    }


def _average_pilot_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    covariance = tf.reduce_mean(
        tf.constant([row["covariance_at_8192"] for row in rows], tf.float64), axis=0
    )
    covariance = 0.5 * (covariance + tf.transpose(covariance))
    eigenvalues, eigenvectors = tf.linalg.eigh(covariance)
    floor = tf.maximum(
        tf.reduce_max(eigenvalues) * tf.constant(1.0e-12, tf.float64),
        tf.constant(1.0e-15, tf.float64),
    )
    covariance = tf.einsum(
        "ij,j,kj->ik", eigenvectors, tf.maximum(eigenvalues, floor), eigenvectors
    )
    return {
        "truth": rows[0]["truth"],
        "mean_estimate": list(
            tf.reduce_mean(tf.constant([row["estimate"] for row in rows], tf.float64), axis=0).numpy()
        ),
        "covariance_at_8192": covariance.numpy().tolist(),
        "mmd_mean_estimate": sum(row["mmd_estimate"] for row in rows) / len(rows),
        "mmd_root_mean_square_standard_error_at_8192": math.sqrt(
            sum(row["mmd_standard_error_at_8192"] ** 2 for row in rows) / len(rows)
        ),
        "mmd_harmonic_degrees_of_freedom_at_8192": len(rows)
        / sum(1.0 / row["mmd_degrees_of_freedom_at_8192"] for row in rows),
        "replication_count": len(rows),
        "max_condition_number": max(row["condition_number"] for row in rows),
        "max_ridge_multiplier": max(row["ridge_multiplier"] for row in rows),
        "replications": rows,
    }


def _probability_record(mask: tf.Tensor) -> dict[str, Any]:
    count = int(tf.size(mask))
    successes = int(tf.reduce_sum(tf.cast(mask, tf.int32)))
    lower, upper = _wilson_interval(successes, count)
    return {
        "success_count": successes,
        "monte_carlo_count": count,
        "estimate": successes / count,
        "wilson_95_lower": lower,
        "wilson_95_upper": upper,
    }


def _simulate_feature_probabilities(
    *,
    pilot: dict[str, Any],
    draw_count: int,
    scenario: Scenario,
    family_index: int,
    monte_carlo_count: int,
) -> dict[str, dict[str, Any]]:
    covariance = tf.constant(pilot["covariance_at_8192"], tf.float64) * (
        MAX_DRAW_COUNT / draw_count
    )
    eigenvalues, eigenvectors = tf.linalg.eigh(covariance)
    factor = eigenvectors @ tf.linalg.diag(tf.sqrt(tf.maximum(eigenvalues, 0.0)))
    truth = tf.constant(pilot["truth"], tf.float64)
    standard_error = tf.sqrt(tf.linalg.diag_part(covariance))
    counts = {"pass": 0, "material": 0, "inconclusive": 0, "coverage": 0}
    remaining = monte_carlo_count
    batch_index = 0
    while remaining:
        count = min(remaining, MONTE_CARLO_BATCH_SIZE)
        seed = _fold(
            _fold(
                _fold(tf.constant(MONTE_CARLO_SEED, tf.int32), family_index),
                draw_count,
            ),
            100 * list(s.name for s in SCENARIOS).index(scenario.name) + batch_index,
        )
        estimates = truth + _joint_normal_draws(seed, factor, tf.constant(count, tf.int32))
        masks = _feature_status_masks(estimates, standard_error, scenario)
        bonferroni_critical = tfp.distributions.Normal(
            tf.constant(0.0, tf.float64), tf.constant(1.0, tf.float64)
        ).quantile(1.0 - FEATURE_ALPHA / (2.0 * FEATURE_COUNT))
        covered = tf.reduce_all(
            tf.abs(estimates - truth) <= bonferroni_critical * standard_error, axis=1
        )
        for name in ("pass", "material", "inconclusive"):
            counts[name] += int(tf.reduce_sum(tf.cast(masks[name], tf.int32)))
        counts["coverage"] += int(tf.reduce_sum(tf.cast(covered, tf.int32)))
        remaining -= count
        batch_index += 1
    result: dict[str, dict[str, Any]] = {}
    for name, successes in counts.items():
        lower, upper = _wilson_interval(successes, monte_carlo_count)
        result[name] = {
            "success_count": successes,
            "monte_carlo_count": monte_carlo_count,
            "estimate": successes / monte_carlo_count,
            "wilson_95_lower": lower,
            "wilson_95_upper": upper,
        }
    return result


def _mmd_probabilities(
    *, pilot: dict[str, Any], draw_count: int, tolerance: float
) -> dict[str, float]:
    scale = math.sqrt(MAX_DRAW_COUNT / draw_count)
    center = pilot["mmd_mean_estimate"]
    se = pilot["mmd_root_mean_square_standard_error_at_8192"] * scale
    df = max(1.01, pilot["mmd_harmonic_degrees_of_freedom_at_8192"] * draw_count / MAX_DRAW_COUNT)
    critical = float(
        tfp.distributions.StudentT(
            df=tf.constant(df, tf.float64),
            loc=tf.constant(0.0, tf.float64),
            scale=tf.constant(1.0, tf.float64),
        ).quantile(1.0 - MMD_ALPHA / 2.0)
    )
    distribution = tfp.distributions.Normal(
        loc=tf.constant(center, tf.float64), scale=tf.constant(se, tf.float64)
    )
    pass_probability = float(distribution.cdf(tolerance - critical * se))
    material_probability = float(1.0 - distribution.cdf(tolerance + critical * se))
    return {
        "center": center,
        "standard_error": se,
        "degrees_of_freedom": df,
        "critical_value": critical,
        "pass_probability": pass_probability,
        "material_probability": material_probability,
        "inconclusive_probability": max(
            0.0, 1.0 - pass_probability - material_probability
        ),
    }


def _operating_rows(
    pilot_by_family: dict[str, dict[str, Any]], monte_carlo_count: int
) -> dict[str, Any]:
    family_indices = {
        family.name: family_index for family_index, family in enumerate(FAMILIES)
    }
    result: dict[str, Any] = {}
    for scenario in SCENARIOS:
        scenario_rows: dict[str, Any] = {}
        for draw_count in DRAW_GRID:
            family_rows: dict[str, Any] = {}
            for family_name, pilot in pilot_by_family.items():
                family_index = family_indices[family_name]
                family = FAMILIES[family_index]
                feature = _simulate_feature_probabilities(
                    pilot=pilot,
                    draw_count=draw_count,
                    scenario=scenario,
                    family_index=family_index,
                    monte_carlo_count=monte_carlo_count,
                )
                tolerance_rows = {}
                for tolerance in MMD_TOLERANCES:
                    mmd = _mmd_probabilities(
                        pilot=pilot, draw_count=draw_count, tolerance=tolerance
                    )
                    pass_bounds = _frechet_bounds(
                        feature["pass"]["estimate"], mmd["pass_probability"]
                    )
                    material_bounds = (
                        max(feature["material"]["estimate"], mmd["material_probability"]),
                        min(
                            1.0,
                            feature["material"]["estimate"]
                            + mmd["material_probability"],
                        ),
                    )
                    tolerance_rows[str(tolerance)] = {
                        "mmd": mmd,
                        "combined_pass_frechet_bounds": list(pass_bounds),
                        "combined_material_frechet_bounds": list(material_bounds),
                    }
                family_rows[family_name] = {
                    "role": family.role,
                    "feature_probabilities": feature,
                    "mmd_by_tolerance": tolerance_rows,
                }
            scenario_rows[str(draw_count)] = family_rows
        result[scenario.name] = scenario_rows
    return result


def _scenario_feasibility(operating: dict[str, Any]) -> dict[str, Any]:
    family_by_name = {family.name: family for family in FAMILIES}
    result: dict[str, Any] = {}
    for scenario in SCENARIOS:
        scenario_result: dict[str, Any] = {}
        for draw_count in DRAW_GRID:
            rows = operating[scenario.name][str(draw_count)]
            tolerance_result = {}
            for tolerance in MMD_TOLERANCES:
                key = str(tolerance)
                failures: list[str] = []
                for family_name, row in rows.items():
                    family = family_by_name[family_name]
                    if family.role == "explanatory":
                        continue
                    feature = row["feature_probabilities"]
                    combined = row["mmd_by_tolerance"][key]
                    if feature["coverage"]["wilson_95_lower"] < MINIMUM_COVERAGE:
                        failures.append(f"{family_name}:coverage")
                    if family.role == "equivalence":
                        if combined["combined_pass_frechet_bounds"][0] < MINIMUM_REQUIRED_DECISION:
                            failures.append(f"{family_name}:required_equivalence")
                        if combined["combined_material_frechet_bounds"][1] > MAXIMUM_FALSE_DECISION:
                            failures.append(f"{family_name}:false_material")
                    else:
                        if combined["combined_material_frechet_bounds"][0] < MINIMUM_REQUIRED_DECISION:
                            failures.append(f"{family_name}:required_material")
                        if combined["combined_pass_frechet_bounds"][1] > MAXIMUM_FALSE_DECISION:
                            failures.append(f"{family_name}:false_equivalence")
                tolerance_result[key] = {
                    "feasible_under_dependence_robust_bounds": not failures,
                    "failed_checks": failures,
                }
            scenario_result[str(draw_count)] = tolerance_result
        result[scenario.name] = scenario_result
    return result


def _resource_projection(phase7: dict[str, Any]) -> dict[str, Any]:
    warm_segment_seconds = []
    for chart in phase7["charts"].values():
        segments = chart["segments"]
        warm_segment_seconds.extend(row["elapsed_seconds"] for row in segments[1:])
    seconds_per_256 = max(warm_segment_seconds)
    per_draw = seconds_per_256 / 256.0
    pilot = _strict_json(TARGET_PILOT_PATH)
    pilot_forecasts = pilot["forecast_provenance"]
    warm_forecast_seconds_per_64 = min(
        row["elapsed_seconds"] for row in pilot_forecasts.values()
    )
    warm_forecast_seconds_per_draw = warm_forecast_seconds_per_64 / 64.0
    compile_overhead_seconds = max(
        row["elapsed_seconds"] for row in pilot_forecasts.values()
    ) - warm_forecast_seconds_per_64
    current = 512
    projections = {}
    for confirmation_draws in DRAW_GRID:
        required_retained_draws = confirmation_draws + 64
        acquired_retained_draws = int(
            256 * math.ceil(required_retained_draws / 256.0)
        )
        additional_per_chart = max(0, acquired_retained_draws - current)
        seconds_per_chart = additional_per_chart * per_draw
        forecast_seconds_both_charts = (
            compile_overhead_seconds
            + 2.0 * confirmation_draws * warm_forecast_seconds_per_draw
        )
        hmc_seconds_both_charts = 2.0 * seconds_per_chart
        projections[str(confirmation_draws)] = {
            "required_total_retained_draws_per_chain": required_retained_draws,
            "segment_rounded_acquired_draws_per_chain": acquired_retained_draws,
            "unused_segment_surplus_draws_per_chain": (
                acquired_retained_draws - required_retained_draws
            ),
            "additional_retained_draws_per_chain_from_512": additional_per_chart,
            "estimated_additional_seconds_per_chart": seconds_per_chart,
            "estimated_additional_gpu_hours_for_both_charts": 2.0
            * seconds_per_chart
            / 3600.0,
            "estimated_forecast_seconds_for_both_charts": forecast_seconds_both_charts,
            "estimated_total_hmc_plus_forecast_gpu_hours": (
                hmc_seconds_both_charts + forecast_seconds_both_charts
            )
            / 3600.0,
        }
    return {
        "basis": (
            "maximum observed second 256-draw warm segment across G/H; linear "
            "HMC projection without compilation or burn-in; 256-draw acquisition "
            "rounding; target-pilot warm per-draw forecast rate plus one observed "
            "compile overhead"
        ),
        "warm_segment_seconds_per_256_draws": seconds_per_256,
        "warm_forecast_seconds_per_64_draws": warm_forecast_seconds_per_64,
        "forecast_compile_overhead_seconds": compile_overhead_seconds,
        "projections": projections,
        "nonclaim": "cost estimate only; no HMC acquisition is authorized",
    }


def run(*, mode: str, output: Path, wall_cap_seconds: float) -> dict[str, Any]:
    if mode not in {"smoke", "material"}:
        raise PreflightError("HMC acquisition and confirmation remain closed")
    if not math.isfinite(wall_cap_seconds) or wall_cap_seconds <= 0.0:
        raise PreflightError("wall cap must be positive and finite")
    bindings = _validate_bindings(require_smoke=mode == "material")
    bandwidths = tf.constant(
        bindings["pilot"]["pooled_calibration"]["bandwidth_candidates"],
        tf.float64,
    )
    if tuple(bandwidths.shape) != (5,):
        raise PreflightError("pilot bandwidth shape drift")
    pilot_replications = (
        PILOT_REPLICATIONS_SMOKE if mode == "smoke" else PILOT_REPLICATIONS_MATERIAL
    )
    monte_carlo_count = (
        MONTE_CARLO_COUNT_SMOKE if mode == "smoke" else MONTE_CARLO_COUNT_MATERIAL
    )
    family_indices = (3,) if mode == "smoke" else tuple(range(len(FAMILIES)))
    started_at = _now()
    started = time.perf_counter()
    pilot_by_family: dict[str, dict[str, Any]] = {}
    for family_index in family_indices:
        family = FAMILIES[family_index]
        rows = []
        for replication in range(pilot_replications):
            left, right, truth = _family_paths(family, family_index, replication)
            _require_gpu(left, right, truth, surface="preflight family generation")
            rows.append(_pilot_one(left, right, truth, bandwidths))
            if time.perf_counter() - started > wall_cap_seconds:
                raise PreflightError("sample-size preflight wall cap exceeded")
        pilot_by_family[family.name] = _average_pilot_rows(rows)
    operating = _operating_rows(pilot_by_family, monte_carlo_count)
    if time.perf_counter() - started > wall_cap_seconds:
        raise PreflightError("sample-size preflight wall cap exceeded")
    feasibility = _scenario_feasibility(operating)
    any_feasible = any(
        row["feasible_under_dependence_robust_bounds"]
        for scenario_rows in feasibility.values()
        for draw_rows in scenario_rows.values()
        for row in draw_rows.values()
    )
    decision = (
        "PHASE8_SAMPLE_SIZE_PREFLIGHT_SMOKE_PASSED_MATERIAL_REQUIRED"
        if mode == "smoke"
        else (
            "PHASE8_SAMPLE_SIZE_PREFLIGHT_FEASIBLE_SCENARIO_DIRECT_VALIDATION_REQUIRED"
            if any_feasible
            else "PHASE8_SAMPLE_SIZE_PREFLIGHT_NO_FEASIBLE_SCENARIO_PHASE9_CLOSED"
        )
    )
    trace_counts = {
        "gaussian_paths": _trace_count(_gaussian_paths),
        "joint_normal_draws": _trace_count(_joint_normal_draws),
        "mean_log_variance_influence": _trace_count(
            predictive._mean_log_variance_influence_xla
        ),
        "long_run_covariance": _trace_count(predictive._long_run_covariance_xla),
        "batch_means": _trace_count(predictive._batch_means_xla),
        "cluster_kernel": _trace_count(predictive._cluster_kernel_xla),
        "mmd_block_interval": _trace_count(predictive._mmd_block_interval_xla),
    }
    if any(value != 1 for value in trace_counts.values()):
        raise PreflightError(f"compiled trace gate failed: {trace_counts}")
    wall_time = time.perf_counter() - started
    payload = {
        "schema": f"bayesfilter.ssl_lstm_neutra.phase8_sample_size_preflight.{mode}.v1",
        "status": "PASSED_EXECUTION",
        "decision": decision,
        "mode": mode,
        "bindings": {
            "target_pilot": {
                "path": TARGET_PILOT_PATH.as_posix(),
                "sha256": TARGET_PILOT_SHA256,
                "decision": bindings["pilot"]["decision"],
            },
            "failed_448": {
                "path": FAILED_448_PATH.as_posix(),
                "sha256": FAILED_448_SHA256,
                "decision": bindings["failed_448"]["decision"],
            },
            "failed_1984": {
                "path": FAILED_1984_PATH.as_posix(),
                "sha256": FAILED_1984_SHA256,
                "decision": bindings["failed_1984"]["decision"],
            },
            "phase7_public_timing": {
                "path": PHASE7_RECEIPT_PATH.as_posix(),
                "sha256": PHASE7_RECEIPT_SHA256,
                "decision": bindings["phase7"]["decision"],
            },
            "preflight_smoke": (
                {
                    "path": SMOKE_RECEIPT_PATH.as_posix(),
                    "sha256": SMOKE_RECEIPT_SHA256,
                    "decision": bindings["smoke"]["decision"],
                }
                if mode == "material"
                else None
            ),
        },
        "configuration": {
            "draw_grid_per_chain": list(DRAW_GRID),
            "synthetic_pilot_shape": [
                CHAIN_COUNT,
                MAX_DRAW_COUNT,
                FORECAST_REPLICATION_COUNT,
                HORIZON,
            ],
            "block_length": BLOCK_LENGTH,
            "feature_alpha": FEATURE_ALPHA,
            "mmd_alpha": MMD_ALPHA,
            "mmd_tolerances": list(MMD_TOLERANCES),
            "scenarios": [asdict(row) for row in SCENARIOS],
            "pilot_seed": list(PILOT_SEED),
            "monte_carlo_seed": list(MONTE_CARLO_SEED),
            "pilot_replications": pilot_replications,
            "monte_carlo_count": monte_carlo_count,
            "targets": {
                "coverage_wilson_lower": MINIMUM_COVERAGE,
                "required_decision_frechet_lower": MINIMUM_REQUIRED_DECISION,
                "false_decision_frechet_upper": MAXIMUM_FALSE_DECISION,
            },
            "margin_selection": None,
            "mmd_tolerance_selection": None,
        },
        "analytical_preflight": _analytical_preflight(bindings["failed_1984"]),
        "families": [asdict(FAMILIES[index]) for index in family_indices],
        "synthetic_pilot_by_family": pilot_by_family,
        "operating_characteristics": operating,
        "scenario_feasibility": feasibility,
        "resource_projection": _resource_projection(bindings["phase7"]),
        "assumptions": [
            "joint feature estimates are asymptotically Gaussian",
            "feature covariance and MMD variance scale as 1/N across the draw grid",
            "MMD effective degrees of freedom scale linearly with complete block count",
            "the 8192-draw synthetic pilot law represents the declared controlled families",
            "feature/MMD joint probabilities use dependence-robust Frechet bounds",
            "simulation feasibility still requires direct finite-sample validation",
        ],
        "compile_trace_counts": trace_counts,
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
            "tensorflow_probability": tfp.__version__,
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "physical_devices": [
                device.name for device in tf.config.list_physical_devices("GPU")
            ],
            "logical_devices": [
                device.name for device in tf.config.list_logical_devices("GPU")
            ],
            "jit_compile": True,
            "dtype": "float64",
            "tf32_enabled": bool(
                tf.config.experimental.tensor_float_32_execution_enabled()
            ),
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
            "feasibility evidence only; no scientific margin or MMD tolerance selected",
            "no retained archive or G/H confirmation forecast read",
            "no HMC acquisition or Phase 9 execution authorized",
            "no posterior truth, predictive equivalence, sampler ranking, model adequacy, or default claim",
        ],
    }
    _write(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "material"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-cap-seconds", type=float, required=True)
    args = parser.parse_args()
    physical = tf.config.list_physical_devices("GPU")
    if not physical:
        raise PreflightError("sample-size preflight requires a visible GPU")
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
                "wall_time_seconds": payload["run_manifest"]["wall_time_seconds"],
                "margin_selection": payload["configuration"]["margin_selection"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
