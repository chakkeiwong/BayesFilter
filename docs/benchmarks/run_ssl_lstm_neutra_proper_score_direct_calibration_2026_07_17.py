#!/usr/bin/env python3
"""Direct controlled calibration of the repaired proper-score decision."""

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
    "docs/plans/bayesfilter-ssl-lstm-neutra-proper-score-direct-"
    "calibration-plan-2026-07-17.md"
)
SCRIPT_PATH = Path(__file__).resolve().relative_to(ROOT)
PREDICTIVE_SOURCE = Path("bayesfilter/inference/predictive_equivalence.py")
SMOKE_RECEIPT_PATH = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/"
    "phase-8-predictive-design/proper-score-direct-calibration-smoke.json"
)
SMOKE_RECEIPT_SHA256 = (
    "7554ac456684e02eb802f60320fb7fda927df5d0159bdbfee29402873159398b"
)

CHAIN_COUNT = 4
FORECAST_REPLICATION_COUNT = 2
HORIZON = 10
FEATURE_COUNT = 20
DRAW_LADDER = (4096, 8192)
HAC_MULTIPLIER = 1.0
RIDGE_LADDER = (0.0,)
CONDITION_NUMBER_MAX = 1.0e8
CONFIDENCE_ALPHA = 0.05
MATERIAL_REPLICATION_COUNT = 256
MATERIAL_BATCH_SIZE = 8
SMOKE_REPLICATION_COUNT = 2
SMOKE_DRAW_COUNT = 256
SMOKE_SEED = (27101, 27102)
MATERIAL_SEEDS = ((27201, 27202), (27301, 27302))
FAMILYWISE_ALPHA = 0.05
REQUIRED_FAMILY_COUNT = 11
OPERATING_CLAIM_COUNT = 4
MAXIMUM_MATERIAL_LOOKS = 2
SIMULTANEOUS_CLAIM_COUNT = (
    REQUIRED_FAMILY_COUNT * OPERATING_CLAIM_COUNT * MAXIMUM_MATERIAL_LOOKS
)
BINOMIAL_TAIL_ALPHA = FAMILYWISE_ALPHA / SIMULTANEOUS_CLAIM_COUNT
MINIMUM_COVERAGE = 0.90
MINIMUM_REQUIRED_DECISION = 0.80
MAXIMUM_FALSE_DECISION = 0.05
MAXIMUM_INVALID = 0.05
NEGLIGIBLE_ANCHOR_LOSS = max(0.5 * 0.05**2, 0.25 * math.log(1.05) ** 2)
MATERIAL_ANCHOR_LOSS = min(
    0.5 * 0.20**2,
    0.25 * math.log(1.25) ** 2,
    0.25 * math.log(0.80) ** 2,
)
ACCEPTABLE_AVERAGE_LOSS = 0.5 * (
    NEGLIGIBLE_ANCHOR_LOSS + MATERIAL_ANCHOR_LOSS
)
ACCEPTABLE_HORIZON_LOSS = ACCEPTABLE_AVERAGE_LOSS
LOSS_MATRICES = tf.stack(
    (
        predictive.proper_score_loss(
            tf.fill([HORIZON], tf.constant(1.0 / HORIZON, tf.float64))
        ).loss_matrix,
        *(
            predictive.horizon_proper_score_loss(HORIZON, horizon).loss_matrix
            for horizon in range(HORIZON)
        ),
    )
)


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
    Family("negligible_mean_persistent_0p05", "equivalence", mean_shift=0.05),
    Family(
        "negligible_variance_persistent_1p05",
        "equivalence",
        variance_ratio=1.05,
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
    Family("material_variance_persistent_1p25", "material", variance_ratio=1.25),
    Family("material_variance_persistent_0p80", "material", variance_ratio=0.80),
    Family(
        "material_variance_local_h1_1p25",
        "material",
        variance_ratio=1.25,
        local_horizon=0,
    ),
    Family("skew_explanatory", "explanatory", skew=0.35),
    Family("dependence_explanatory", "explanatory", horizon_rho=0.9),
)


class CalibrationError(RuntimeError):
    """Raised when the direct calibration contract fails closed."""


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

    payload = json.loads(
        _absolute(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=reject,
    )
    if not isinstance(payload, dict):
        raise CalibrationError(f"expected JSON object: {path}")
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


def _trace_count(program: Any) -> int | None:
    method = getattr(program, "experimental_get_tracing_count", None)
    return None if method is None else int(method())


def _require_gpu(*tensors: tf.Tensor, surface: str) -> None:
    devices = [str(tensor.device) for tensor in tensors]
    if not devices or any("GPU:" not in device for device in devices):
        raise CalibrationError(f"{surface} outputs are not GPU resident: {devices}")


def _fold(seed: tf.Tensor, value: int | tf.Tensor) -> tf.Tensor:
    return tf.random.experimental.stateless_fold_in(
        seed, tf.cast(value, tf.int32), alg="philox"
    )


def _gaussian_batch_kernel(
    seed: tf.Tensor,
    phi: tf.Tensor,
    rho: tf.Tensor,
    draw_count: int,
    batch_size: int,
) -> tf.Tensor:
    index = tf.range(HORIZON, dtype=tf.float64)
    horizon_covariance = tf.pow(rho, tf.abs(index[:, None] - index[None, :]))
    horizon_factor = tf.linalg.cholesky(horizon_covariance)

    draw_noise = tf.random.stateless_normal(
        [batch_size, CHAIN_COUNT, draw_count, HORIZON],
        _fold(seed, 1),
        dtype=tf.float64,
        alg="philox",
    ) @ tf.transpose(horizon_factor)
    initial = tf.random.stateless_normal(
        [batch_size, CHAIN_COUNT, HORIZON],
        _fold(seed, 2),
        dtype=tf.float64,
        alg="philox",
    ) @ tf.transpose(horizon_factor)
    innovations = tf.transpose(draw_noise[:, :, 1:, :], [2, 0, 1, 3])
    scanned = tf.scan(
        lambda previous, innovation: phi * previous
        + tf.sqrt(1.0 - tf.square(phi)) * innovation,
        innovations,
        initializer=initial,
    )
    clusters = tf.concat(
        (initial[:, :, None, :], tf.transpose(scanned, [1, 2, 0, 3])), axis=2
    )
    replication_noise = tf.random.stateless_normal(
        [
            batch_size,
            CHAIN_COUNT,
            draw_count,
            FORECAST_REPLICATION_COUNT,
            HORIZON,
        ],
        _fold(seed, 3),
        dtype=tf.float64,
        alg="philox",
    ) @ tf.transpose(horizon_factor)
    return (
        tf.sqrt(tf.constant(0.65, tf.float64)) * clusters[:, :, :, None, :]
        + tf.sqrt(tf.constant(0.35, tf.float64)) * replication_noise
    )


def _feature_batch_kernel(paths: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    batch_size, chain_count, draw_count, replication_count, horizon = paths.shape
    path_count = chain_count * draw_count * replication_count
    means = tf.reduce_mean(paths, axis=[1, 2, 3])
    centered = paths - means[:, None, None, None, :]
    mean_influence = tf.reduce_mean(centered, axis=3)
    second_moment = tf.reduce_mean(tf.square(centered), axis=[1, 2, 3])
    variances = second_moment * tf.cast(path_count, tf.float64) / tf.cast(
        path_count - 1, tf.float64
    )
    log_variance_influence = (
        tf.reduce_mean(tf.square(centered), axis=3)
        / second_moment[:, None, None, :]
        - 1.0
    )
    estimate = tf.concat((means, tf.math.log(variances)), axis=1)
    influence = tf.concat((mean_influence, log_variance_influence), axis=3)
    return estimate, influence


_COMPILED: dict[tuple[str, int, int], Any] = {}


def _compiled(name: str, *, draw_count: int, batch_size: int, function: Any) -> Any:
    key = (name, draw_count, batch_size)
    if key not in _COMPILED:
        signature = (
            [tf.TensorSpec([2], tf.int32), tf.TensorSpec([], tf.float64), tf.TensorSpec([], tf.float64)]
            if name == "paths"
            else [
                tf.TensorSpec(
                    [
                        batch_size,
                        CHAIN_COUNT,
                        draw_count,
                        FORECAST_REPLICATION_COUNT,
                        HORIZON,
                    ],
                    tf.float64,
                )
            ]
        )
        if name == "paths":
            def bound(seed: tf.Tensor, phi: tf.Tensor, rho: tf.Tensor) -> tf.Tensor:
                return function(seed, phi, rho, draw_count, batch_size)
        else:
            bound = function
        _COMPILED[key] = tf.function(
            bound, input_signature=signature, autograph=False, jit_compile=True
        )
    return _COMPILED[key]


def _family_truth(family: Family) -> tf.Tensor:
    shift = tf.zeros([HORIZON], tf.float64)
    log_ratio = tf.zeros([HORIZON], tf.float64)
    if family.mean_shift:
        if family.local_horizon is None:
            shift = tf.fill([HORIZON], tf.constant(family.mean_shift, tf.float64))
        else:
            shift = tf.tensor_scatter_nd_update(
                shift, [[family.local_horizon]], [family.mean_shift]
            )
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
        log_ratio = tf.math.log(ratios)
    return tf.concat((-shift, -log_ratio), axis=0)


def _apply_family(paths: tf.Tensor, family: Family) -> tf.Tensor:
    result = paths
    if family.mean_shift:
        shift = -_family_truth(family)[:HORIZON]
        result = result + shift[None, None, None, None, :]
    if family.variance_ratio != 1.0:
        ratios = tf.exp(-_family_truth(family)[HORIZON:])
        result = result * tf.sqrt(ratios)[None, None, None, None, :]
    if family.skew:
        coefficient = tf.constant(family.skew, tf.float64)
        result = (result + coefficient * (tf.square(result) - 1.0)) / tf.sqrt(
            1.0 + 2.0 * tf.square(coefficient)
        )
    return result


def _exact_one_sided_interval(successes: int, count: int) -> tuple[float, float]:
    if not 0 <= successes <= count or count < 1:
        raise CalibrationError("invalid binomial count")
    lower = (
        0.0
        if successes == 0
        else float(
            tfp.distributions.Beta(
                tf.constant(float(successes), tf.float64),
                tf.constant(float(count - successes + 1), tf.float64),
            ).quantile(BINOMIAL_TAIL_ALPHA)
        )
    )
    upper = (
        1.0
        if successes == count
        else float(
            tfp.distributions.Beta(
                tf.constant(float(successes + 1), tf.float64),
                tf.constant(float(count - successes), tf.float64),
            ).quantile(1.0 - BINOMIAL_TAIL_ALPHA)
        )
    )
    return lower, upper


def _probability_record(successes: int, count: int) -> dict[str, Any]:
    lower, upper = _exact_one_sided_interval(successes, count)
    return {
        "success_count": successes,
        "replication_count": count,
        "estimate": successes / count,
        "simultaneous_lower": lower,
        "simultaneous_upper": upper,
        "one_sided_tail_alpha": BINOMIAL_TAIL_ALPHA,
    }


def _family_batch(
    family: Family,
    *,
    family_index: int,
    draw_count: int,
    base_seed: tuple[int, int],
    batch_start: int,
    batch_size: int,
) -> dict[str, tf.Tensor]:
    root = tf.constant(base_seed, tf.int32)
    batch_seed = _fold(_fold(root, family_index), batch_start)
    path_program = _compiled(
        "paths", draw_count=draw_count, batch_size=batch_size, function=_gaussian_batch_kernel
    )
    feature_program = _compiled(
        "features", draw_count=draw_count, batch_size=batch_size, function=_feature_batch_kernel
    )
    left = path_program(
        _fold(batch_seed, 11),
        tf.constant(family.phi, tf.float64),
        tf.constant(0.7, tf.float64),
    )
    right = path_program(
        _fold(batch_seed, 22),
        tf.constant(family.phi, tf.float64),
        tf.constant(family.horizon_rho, tf.float64),
    )
    right = _apply_family(right, family)
    left_estimate, left_influence = feature_program(left)
    right_estimate, right_influence = feature_program(right)
    estimate = left_estimate - right_estimate
    influence = tf.concat((2.0 * left_influence, -2.0 * right_influence), axis=1)
    hac = predictive.batched_chain_bartlett_long_run_covariance(
        influence,
        bandwidth_multiplier=HAC_MULTIPLIER,
        ridge_ladder=RIDGE_LADDER,
        condition_number_max=CONDITION_NUMBER_MAX,
    )
    bounds = predictive.batched_quadratic_loss_confidence_bounds(
        estimate,
        hac.regularized_covariance,
        LOSS_MATRICES,
        alpha=CONFIDENCE_ALPHA,
    )
    valid = tf.logical_and(hac.inference_admissible, bounds.inference_admissible)
    average_upper = bounds.upper_bound[:, 0]
    horizon_upper = bounds.upper_bound[:, 1:]
    average_lower = bounds.lower_bound[:, 0]
    horizon_lower = bounds.lower_bound[:, 1:]
    passing = tf.logical_and(
        valid,
        tf.logical_and(
            average_upper < ACCEPTABLE_AVERAGE_LOSS,
            tf.reduce_all(horizon_upper < ACCEPTABLE_HORIZON_LOSS, axis=1),
        ),
    )
    material = tf.logical_and(
        valid,
        tf.logical_or(
            average_lower > ACCEPTABLE_AVERAGE_LOSS,
            tf.reduce_any(horizon_lower > ACCEPTABLE_HORIZON_LOSS, axis=1),
        ),
    )
    truth = _family_truth(family)
    centered = estimate - truth[None, :]
    quadratic = tf.einsum("bi,bij,bj->b", centered, hac.precision, centered)
    covered = tf.logical_and(
        valid,
        quadratic <= bounds.confidence_radius_squared,
    )
    _require_gpu(
        estimate,
        hac.regularized_covariance,
        bounds.upper_bound,
        surface="proper-score direct calibration",
    )
    return {
        "estimate": estimate,
        "valid": valid,
        "pass": passing,
        "material": material,
        "covered": covered,
        "condition_number": hac.condition_number,
        "point_loss": bounds.point_loss,
        "lower_bound": bounds.lower_bound,
        "upper_bound": bounds.upper_bound,
        "lower_kkt": bounds.lower_kkt_residual,
        "upper_kkt": bounds.upper_kkt_residual,
        "bandwidth": tf.constant(hac.bandwidth, tf.int32),
    }


def _summarize_family(
    family: Family,
    batches: list[dict[str, tf.Tensor]],
    replication_count: int,
) -> dict[str, Any]:
    joined = {
        key: tf.concat([batch[key] for batch in batches], axis=0)
        for key in (
            "estimate",
            "valid",
            "pass",
            "material",
            "covered",
            "condition_number",
            "point_loss",
            "lower_bound",
            "upper_bound",
            "lower_kkt",
            "upper_kkt",
        )
    }
    valid_count = int(tf.reduce_sum(tf.cast(joined["valid"], tf.int32)))
    invalid_count = replication_count - valid_count
    coverage_count = int(tf.reduce_sum(tf.cast(joined["covered"], tf.int32)))
    pass_count = int(tf.reduce_sum(tf.cast(joined["pass"], tf.int32)))
    material_count = int(tf.reduce_sum(tf.cast(joined["material"], tf.int32)))
    if family.role == "equivalence":
        required_count = pass_count
        false_count = material_count
    elif family.role == "material":
        required_count = material_count
        false_count = pass_count
    else:
        required_count = 0
        false_count = 0
    finite_condition_numbers = tf.boolean_mask(
        joined["condition_number"], tf.math.is_finite(joined["condition_number"])
    )
    maximum_condition_number = (
        None
        if int(tf.size(finite_condition_numbers)) == 0
        else float(tf.reduce_max(finite_condition_numbers))
    )

    def finite_maximum(values: tf.Tensor) -> float | None:
        finite = tf.boolean_mask(values, tf.math.is_finite(values))
        return None if int(tf.size(finite)) == 0 else float(tf.reduce_max(finite))

    result = {
        "family": asdict(family),
        "truth": _family_truth(family).numpy().tolist(),
        "counts": {
            "pass": pass_count,
            "material_difference": material_count,
            "inconclusive": replication_count - invalid_count - pass_count - material_count,
            "invalid": invalid_count,
        },
        "coverage": _probability_record(coverage_count, replication_count),
        "required_decision": _probability_record(required_count, replication_count),
        "false_decision": _probability_record(false_count, replication_count),
        "invalid_procedure": _probability_record(invalid_count, replication_count),
        "descriptive": {
            "maximum_finite_condition_number": maximum_condition_number,
            "maximum_finite_lower_kkt_residual": finite_maximum(joined["lower_kkt"]),
            "maximum_finite_upper_kkt_residual": finite_maximum(joined["upper_kkt"]),
            "mean_average_point_loss": float(tf.reduce_mean(joined["point_loss"][:, 0])),
            "maximum_horizon_point_loss": float(tf.reduce_max(joined["point_loss"][:, 1:])),
            "mean_estimate": tf.reduce_mean(joined["estimate"], axis=0).numpy().tolist(),
        },
    }
    if family.role != "explanatory":
        result["gate"] = {
            "coverage": result["coverage"]["simultaneous_lower"] >= MINIMUM_COVERAGE,
            "required_decision": (
                result["required_decision"]["simultaneous_lower"]
                >= MINIMUM_REQUIRED_DECISION
            ),
            "false_decision": (
                result["false_decision"]["simultaneous_upper"]
                <= MAXIMUM_FALSE_DECISION
            ),
            "invalid_procedure": (
                result["invalid_procedure"]["simultaneous_upper"] <= MAXIMUM_INVALID
            ),
        }
        result["gate"]["passed"] = all(result["gate"].values())
    else:
        result["gate"] = None
    return result


def _run_rung(
    *,
    draw_count: int,
    replication_count: int,
    batch_size: int,
    seed: tuple[int, int],
    families: tuple[Family, ...],
    wall_deadline: float,
) -> dict[str, Any]:
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    for family_index, family in enumerate(families):
        batches: list[dict[str, tf.Tensor]] = []
        for batch_start in range(0, replication_count, batch_size):
            if time.monotonic() >= wall_deadline:
                raise CalibrationError("wall cap exhausted during direct calibration")
            current_batch = min(batch_size, replication_count - batch_start)
            batches.append(
                _family_batch(
                    family,
                    family_index=family_index,
                    draw_count=draw_count,
                    base_seed=seed,
                    batch_start=batch_start,
                    batch_size=current_batch,
                )
            )
        rows.append(_summarize_family(family, batches, replication_count))
    required_rows = [row for row in rows if row["family"]["role"] != "explanatory"]
    return {
        "draw_count_per_chain": draw_count,
        "replication_count_per_family": replication_count,
        "batch_size": batch_size,
        "seed": list(seed),
        "bandwidth": predictive.growing_hac_bandwidth(
            draw_count, multiplier=HAC_MULTIPLIER
        ),
        "families": rows,
        "gate_passed": all(row["gate"]["passed"] for row in required_rows),
        "wall_seconds": time.monotonic() - started,
    }


def _validate_smoke_receipt() -> dict[str, Any]:
    if _sha256(SMOKE_RECEIPT_PATH) != SMOKE_RECEIPT_SHA256:
        raise CalibrationError("smoke receipt identity drift")
    payload = _strict_json(SMOKE_RECEIPT_PATH)
    if (
        payload.get("decision") != "DIRECT_CALIBRATION_SMOKE_PASSED_MATERIAL_REQUIRED"
        or payload.get("configuration", {}).get("mode") != "smoke"
        or payload.get("claim_boundary", {}).get("statistical_evidence") is not False
    ):
        raise CalibrationError("smoke receipt contract drift")
    return payload


def run(*, mode: str, output: Path, wall_cap_seconds: float) -> dict[str, Any]:
    if mode not in {"smoke", "material"}:
        raise CalibrationError("HMC acquisition and confirmation remain closed")
    if wall_cap_seconds <= 0.0 or not math.isfinite(wall_cap_seconds):
        raise CalibrationError("wall_cap_seconds must be finite and positive")
    if _absolute(output).exists():
        raise CalibrationError(f"refusing to overwrite receipt: {output}")
    if not tf.config.list_physical_devices("GPU"):
        raise CalibrationError("trusted GPU is required")
    started_wall = time.monotonic()
    deadline = started_wall + wall_cap_seconds
    started_at = _now()
    smoke_binding = None
    if mode == "material":
        smoke_binding = {
            "path": str(SMOKE_RECEIPT_PATH),
            "sha256": _sha256(SMOKE_RECEIPT_PATH),
            "decision": _validate_smoke_receipt()["decision"],
        }
        families = FAMILIES
        ladder = DRAW_LADDER
        replication_count = MATERIAL_REPLICATION_COUNT
        batch_size = MATERIAL_BATCH_SIZE
    else:
        families = (FAMILIES[1], FAMILIES[6])
        ladder = (SMOKE_DRAW_COUNT,)
        replication_count = SMOKE_REPLICATION_COUNT
        batch_size = SMOKE_REPLICATION_COUNT

    rungs: list[dict[str, Any]] = []
    for rung_index, draw_count in enumerate(ladder):
        seed = SMOKE_SEED if mode == "smoke" else MATERIAL_SEEDS[rung_index]
        rung = _run_rung(
            draw_count=draw_count,
            replication_count=replication_count,
            batch_size=batch_size,
            seed=seed,
            families=families,
            wall_deadline=deadline,
        )
        rungs.append(rung)
        if mode == "material" and rung["gate_passed"]:
            break

    if mode == "smoke":
        decision = "DIRECT_CALIBRATION_SMOKE_PASSED_MATERIAL_REQUIRED"
    elif rungs[-1]["gate_passed"]:
        decision = "DIRECT_CALIBRATION_PASSED_CONTROLLED_DESIGN_ONLY"
    else:
        decision = "DIRECT_CALIBRATION_CANDIDATE_FAILED_REPAIR_REQUIRED"

    logical_devices = [device.name for device in tf.config.list_logical_devices()]
    payload = {
        "schema": "bayesfilter.ssl_lstm.proper_score_direct_calibration.v1",
        "decision": decision,
        "started_at_utc": started_at,
        "finished_at_utc": _now(),
        "configuration": {
            "mode": mode,
            "chain_count": CHAIN_COUNT,
            "forecast_replication_count": FORECAST_REPLICATION_COUNT,
            "horizon": HORIZON,
            "draw_ladder": list(ladder),
            "material_replication_count": MATERIAL_REPLICATION_COUNT,
            "material_batch_size": MATERIAL_BATCH_SIZE,
            "confidence_alpha": CONFIDENCE_ALPHA,
            "hac_multiplier": HAC_MULTIPLIER,
            "ridge_ladder": list(RIDGE_LADDER),
            "condition_number_max": CONDITION_NUMBER_MAX,
            "negligible_anchor_loss": NEGLIGIBLE_ANCHOR_LOSS,
            "material_anchor_loss": MATERIAL_ANCHOR_LOSS,
            "acceptable_average_loss": ACCEPTABLE_AVERAGE_LOSS,
            "acceptable_horizon_loss": ACCEPTABLE_HORIZON_LOSS,
            "simultaneous_claim_count": SIMULTANEOUS_CLAIM_COUNT,
            "binomial_tail_alpha": BINOMIAL_TAIL_ALPHA,
            "minimum_coverage": MINIMUM_COVERAGE,
            "minimum_required_decision": MINIMUM_REQUIRED_DECISION,
            "maximum_false_decision": MAXIMUM_FALSE_DECISION,
            "maximum_invalid": MAXIMUM_INVALID,
        },
        "smoke_binding": smoke_binding,
        "rungs": rungs,
        "sequential_stop": {
            "stopped_after_first_passing_rung": (
                mode == "material" and len(rungs) == 1 and rungs[0]["gate_passed"]
            ),
            "maximum_rungs": len(ladder),
        },
        "claim_boundary": {
            "statistical_evidence": mode == "material",
            "controlled_laws_only": True,
            "mmd_role": "explanatory_only_not_computed",
            "hmc_authorized": False,
            "confirmation_forecast_bank_opened": False,
            "g_h_predictive_difference_computed": False,
            "posterior_correctness_concluded": False,
            "default_readiness_concluded": False,
        },
        "run_manifest": {
            "git_commit": _git("rev-parse", "HEAD").strip(),
            "git_status_porcelain": _git("status", "--short").splitlines(),
            "command": " ".join(shlex.quote(argument) for argument in sys.argv),
            "python": sys.version,
            "platform": platform.platform(),
            "tensorflow": tf.__version__,
            "tensorflow_probability": tfp.__version__,
            "physical_devices": [device.name for device in tf.config.list_physical_devices()],
            "logical_devices": logical_devices,
            "float_type": "float64",
            "jit_compile": True,
            "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
            "wall_cap_seconds": wall_cap_seconds,
            "wall_seconds": time.monotonic() - started_wall,
            "plan_path": str(PLAN_PATH),
            "output_path": str(output),
            "source_hashes": {
                str(PLAN_PATH): _sha256(PLAN_PATH),
                str(SCRIPT_PATH): _sha256(SCRIPT_PATH),
                str(PREDICTIVE_SOURCE): _sha256(PREDICTIVE_SOURCE),
            },
            "compiled_trace_counts": {
                "|".join(map(str, key)): _trace_count(program)
                for key, program in _COMPILED.items()
            },
        },
    }
    _write(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "material"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-cap-seconds", type=float, required=True)
    arguments = parser.parse_args()
    payload = run(
        mode=arguments.mode,
        output=arguments.output,
        wall_cap_seconds=arguments.wall_cap_seconds,
    )
    print(json.dumps({"decision": payload["decision"], "output": str(arguments.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
