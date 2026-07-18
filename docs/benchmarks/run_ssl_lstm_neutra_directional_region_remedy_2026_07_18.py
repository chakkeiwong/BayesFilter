#!/usr/bin/env python3
"""Controlled split-region/HAC/Rao-Blackwell remedy for SSL-LSTM validation."""

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
    "docs/plans/bayesfilter-ssl-lstm-neutra-directional-region-remedy-plan-"
    "2026-07-18.md"
)
SCRIPT_PATH = Path(__file__).resolve().relative_to(ROOT)
PREDICTIVE_SOURCE = Path("bayesfilter/inference/predictive_equivalence.py")
DEVELOPMENT_RECEIPT = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/"
    "phase-8-predictive-design/directional-region-remedy/development.json"
)
CAPACITY_RECEIPT = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/"
    "phase-8-predictive-design/directional-region-remedy/capacity.json"
)

CHAIN_COUNT = 4
FORECAST_REPLICATION_COUNT = 2
HORIZON = 10
FEATURE_COUNT = 20
DRAW_COUNT = 8192
CAPACITY_DRAW_LADDER = (12288, 16384)
SMOKE_DRAW_COUNT = 256
SMOKE_REPLICATION_COUNT = 2
DEVELOPMENT_REPLICATION_COUNT = 96
CAPACITY_REPLICATION_COUNT = 96
AUDIT_REPLICATION_COUNT = 1536
DEVELOPMENT_BATCH_SIZE = 8
AUDIT_BATCH_SIZE = 8
CAPACITY_MINIMUM_POOLED_COVERAGE = 0.93
CAPACITY_MAXIMUM_POOLED_COVERAGE = 0.97
CAPACITY_MINIMUM_REQUIRED_DECISION = 0.85
CAPACITY_MAXIMUM_FALSE_DECISION = 0.02
AVERAGE_ALPHA = 0.025
HORIZON_ALPHA = 0.0025
FAMILYWISE_REGION_ALPHA = 0.05
HAC_CANDIDATES = (1.0, 1.5, 2.0, 3.0)
RIDGE_LADDER = (0.0,)
CONDITION_NUMBER_MAX = 1.0e8
FAMILYWISE_OPERATING_ALPHA = 0.05
MINIMUM_COVERAGE = 0.90
MINIMUM_REQUIRED_DECISION = 0.80
MAXIMUM_FALSE_DECISION = 0.05
MAXIMUM_INVALID = 0.05
MAXIMUM_BOUNDARY_DECISION = 0.05
NEGLIGIBLE_ANCHOR_LOSS = max(0.5 * 0.05**2, 0.25 * math.log(1.05) ** 2)
MATERIAL_ANCHOR_LOSS = min(
    0.5 * 0.20**2,
    0.25 * math.log(1.25) ** 2,
    0.25 * math.log(0.80) ** 2,
)
ACCEPTABLE_LOSS = 0.5 * (NEGLIGIBLE_ANCHOR_LOSS + MATERIAL_ANCHOR_LOSS)
BOUNDARY_MEAN = math.sqrt(2.0 * ACCEPTABLE_LOSS)
BOUNDARY_VARIANCE_UP = math.exp(2.0 * math.sqrt(ACCEPTABLE_LOSS))
BOUNDARY_VARIANCE_DOWN = math.exp(-2.0 * math.sqrt(ACCEPTABLE_LOSS))
BOUNDARY_EPSILON = 0.002
AVERAGE_LOSS = predictive.proper_score_loss(
    tf.fill([HORIZON], tf.constant(1.0 / HORIZON, tf.float64))
)
FULL_LOSS_MATRICES = tf.stack(
    (
        AVERAGE_LOSS.loss_matrix,
        *(
            predictive.horizon_proper_score_loss(HORIZON, horizon).loss_matrix
            for horizon in range(HORIZON)
        ),
    )
)
SMOKE_SEED = (28101, 28102)
DEVELOPMENT_SEED = (28201, 28202)
AUDIT_SEED = (28301, 28302)


@dataclass(frozen=True)
class Family:
    name: str
    role: Literal[
        "equivalence",
        "material",
        "guard_equivalence",
        "guard_material",
        "boundary",
        "explanatory",
    ]
    phi: float = 0.6
    horizon_rho: float = 0.7
    mean_shift: float = 0.0
    variance_ratio: float = 1.0
    local_horizon: int | None = None
    skew: float = 0.0
    heavy_tail: bool = False


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
    Family(
        "boundary_mean_local_h1_exact",
        "boundary",
        mean_shift=BOUNDARY_MEAN,
        local_horizon=0,
    ),
    Family(
        "boundary_mean_local_h1_inside",
        "guard_equivalence",
        mean_shift=BOUNDARY_MEAN - BOUNDARY_EPSILON,
        local_horizon=0,
    ),
    Family(
        "boundary_mean_local_h1_outside",
        "guard_material",
        mean_shift=BOUNDARY_MEAN + BOUNDARY_EPSILON,
        local_horizon=0,
    ),
    Family(
        "boundary_variance_local_h1_exact_up",
        "boundary",
        variance_ratio=BOUNDARY_VARIANCE_UP,
        local_horizon=0,
    ),
    Family(
        "boundary_variance_local_h1_exact_down",
        "boundary",
        variance_ratio=BOUNDARY_VARIANCE_DOWN,
        local_horizon=0,
    ),
    Family(
        "boundary_mean_persistent_exact",
        "boundary",
        mean_shift=BOUNDARY_MEAN,
    ),
    Family(
        "boundary_mean_persistent_inside",
        "guard_equivalence",
        mean_shift=BOUNDARY_MEAN - BOUNDARY_EPSILON,
    ),
    Family(
        "boundary_mean_persistent_outside",
        "guard_material",
        mean_shift=BOUNDARY_MEAN + BOUNDARY_EPSILON,
    ),
    Family(
        "boundary_variance_persistent_exact_up",
        "boundary",
        variance_ratio=BOUNDARY_VARIANCE_UP,
    ),
    Family(
        "boundary_variance_local_h1_inside_up",
        "guard_equivalence",
        variance_ratio=math.exp(2.0 * math.sqrt(ACCEPTABLE_LOSS) - BOUNDARY_EPSILON),
        local_horizon=0,
    ),
    Family(
        "boundary_variance_local_h1_outside_up",
        "guard_material",
        variance_ratio=math.exp(2.0 * math.sqrt(ACCEPTABLE_LOSS) + BOUNDARY_EPSILON),
        local_horizon=0,
    ),
    Family("skew_explanatory", "explanatory", skew=0.35),
    Family("heavy_tail_explanatory", "explanatory", heavy_tail=True),
    Family("dependence_explanatory", "explanatory", horizon_rho=0.9),
)
PRIMARY_FAMILIES = tuple(
    family for family in FAMILIES if family.role in {"equivalence", "material"}
)
BOUNDARY_FAMILIES = tuple(family for family in FAMILIES if family.role == "boundary")
GUARD_FAMILIES = tuple(family for family in FAMILIES if family.role.startswith("guard_"))
AUDIT_OPERATING_CLAIMS = (
    len(PRIMARY_FAMILIES) * 4
    + len(BOUNDARY_FAMILIES) * 3
    + len(GUARD_FAMILIES) * 3
)
BINOMIAL_TAIL_ALPHA = FAMILYWISE_OPERATING_ALPHA / AUDIT_OPERATING_CLAIMS


class RemedyError(RuntimeError):
    """Raised when the controlled remedy contract fails closed."""


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(_absolute(path).read_bytes()).hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise RemedyError(f"duplicate JSON key {key!r}: {path}")
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise RemedyError(f"nonfinite JSON constant {value!r}: {path}")

    payload = json.loads(
        _absolute(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=reject,
    )
    if not isinstance(payload, dict):
        raise RemedyError(f"expected JSON object: {path}")
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
        raise RemedyError(f"refusing to overwrite receipt: {path}")
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


def _require_gpu(*tensors: tf.Tensor, surface: str) -> None:
    devices = [str(tensor.device) for tensor in tensors]
    if not devices or any("GPU:" not in device for device in devices):
        raise RemedyError(f"{surface} outputs are not GPU resident: {devices}")


def _fold(seed: tf.Tensor, value: int | tf.Tensor) -> tf.Tensor:
    return tf.random.experimental.stateless_fold_in(
        seed, tf.cast(value, tf.int32), alg="philox"
    )


def _gaussian_components_kernel(
    seed: tf.Tensor,
    phi: tf.Tensor,
    rho: tf.Tensor,
    draw_count: int,
    batch_size: int,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
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
    conditional_means = (
        tf.sqrt(tf.constant(0.65, tf.float64))
        * clusters[:, :, :, None, :]
        * tf.ones([FORECAST_REPLICATION_COUNT], tf.float64)[None, None, None, :, None]
    )
    conditional_variances = tf.fill(
        tf.shape(conditional_means), tf.constant(0.35, tf.float64)
    )
    replication_noise = tf.random.stateless_normal(
        tf.shape(conditional_means),
        _fold(seed, 3),
        dtype=tf.float64,
        alg="philox",
    ) @ tf.transpose(horizon_factor)
    paths = conditional_means + tf.sqrt(conditional_variances) * replication_noise
    return paths, conditional_means, conditional_variances


def _path_feature_kernel(paths: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    batch_size, chain_count, draw_count, replication_count, _ = paths.shape
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
    return tf.concat((means, tf.math.log(variances)), axis=1), tf.concat(
        (mean_influence, log_variance_influence), axis=3
    )


def _conditional_feature_kernel(
    conditional_means: tf.Tensor, conditional_variances: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    means = tf.reduce_mean(conditional_means, axis=[1, 2, 3])
    second_moments = tf.reduce_mean(
        conditional_variances + tf.square(conditional_means), axis=[1, 2, 3]
    )
    variances = second_moments - tf.square(means)
    mean_influence = tf.reduce_mean(
        conditional_means - means[:, None, None, None, :], axis=3
    )
    variance_contribution = conditional_variances + tf.square(
        conditional_means - means[:, None, None, None, :]
    )
    log_variance_influence = (
        tf.reduce_mean(variance_contribution, axis=3)
        / variances[:, None, None, :]
        - 1.0
    )
    return tf.concat((means, tf.math.log(variances)), axis=1), tf.concat(
        (mean_influence, log_variance_influence), axis=3
    )


_COMPILED: dict[tuple[str, int, int], Any] = {}


def _compiled(name: str, draw_count: int, batch_size: int) -> Any:
    key = (name, draw_count, batch_size)
    if key in _COMPILED:
        return _COMPILED[key]
    if name == "components":
        def bound(seed: tf.Tensor, phi: tf.Tensor, rho: tf.Tensor):
            return _gaussian_components_kernel(seed, phi, rho, draw_count, batch_size)

        signature = [
            tf.TensorSpec([2], tf.int32),
            tf.TensorSpec([], tf.float64),
            tf.TensorSpec([], tf.float64),
        ]
    elif name == "path_features":
        bound = _path_feature_kernel
        signature = [
            tf.TensorSpec(
                [batch_size, CHAIN_COUNT, draw_count, FORECAST_REPLICATION_COUNT, HORIZON],
                tf.float64,
            )
        ]
    elif name == "conditional_features":
        bound = _conditional_feature_kernel
        signature = [
            tf.TensorSpec(
                [batch_size, CHAIN_COUNT, draw_count, FORECAST_REPLICATION_COUNT, HORIZON],
                tf.float64,
            ),
            tf.TensorSpec(
                [batch_size, CHAIN_COUNT, draw_count, FORECAST_REPLICATION_COUNT, HORIZON],
                tf.float64,
            ),
        ]
    else:
        raise RemedyError(f"unknown compiled surface: {name}")
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
        ratios = tf.ones([HORIZON], tf.float64)
        if family.local_horizon is None:
            ratios = tf.fill(
                [HORIZON], tf.constant(family.variance_ratio, tf.float64)
            )
        else:
            ratios = tf.tensor_scatter_nd_update(
                ratios, [[family.local_horizon]], [family.variance_ratio]
            )
        log_ratio = tf.math.log(ratios)
    return tf.concat((-shift, -log_ratio), axis=0)


def _apply_family(
    paths: tf.Tensor,
    conditional_means: tf.Tensor,
    conditional_variances: tf.Tensor,
    family: Family,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    truth = _family_truth(family)
    shift = -truth[:HORIZON]
    ratios = tf.exp(-truth[HORIZON:])
    scale = tf.sqrt(ratios)[None, None, None, None, :]
    means = conditional_means * scale + shift[None, None, None, None, :]
    variances = conditional_variances * ratios[None, None, None, None, :]
    transformed_paths = paths * scale + shift[None, None, None, None, :]
    if family.skew:
        coefficient = tf.constant(family.skew, tf.float64)
        transformed_paths = (
            transformed_paths
            + coefficient * (tf.square(transformed_paths) - 1.0)
        ) / tf.sqrt(1.0 + 2.0 * tf.square(coefficient))
    if family.heavy_tail:
        normalizer = tf.sqrt(
            tf.pow(tf.constant(2.0, tf.float64), 1.25)
            * tf.exp(tf.math.lgamma(tf.constant(1.75, tf.float64)))
            / tf.sqrt(tf.constant(math.pi, tf.float64))
        )
        transformed_paths = (
            tf.sign(transformed_paths)
            * tf.pow(tf.abs(transformed_paths), tf.constant(1.25, tf.float64))
            / normalizer
        )
    return transformed_paths, means, variances


def _exact_one_sided_interval(
    successes: int, count: int, *, tail_alpha: float
) -> tuple[float, float]:
    if not 0 <= successes <= count or count < 1:
        raise RemedyError("invalid binomial count")
    lower = 0.0 if successes == 0 else float(
        tfp.distributions.Beta(
            tf.constant(float(successes), tf.float64),
            tf.constant(float(count - successes + 1), tf.float64),
        ).quantile(tf.constant(tail_alpha, tf.float64))
    )
    upper = 1.0 if successes == count else float(
        tfp.distributions.Beta(
            tf.constant(float(successes + 1), tf.float64),
            tf.constant(float(count - successes), tf.float64),
        ).quantile(tf.constant(1.0 - tail_alpha, tf.float64))
    )
    return lower, upper


def _probability_record(
    successes: int, count: int, *, tail_alpha: float
) -> dict[str, Any]:
    lower, upper = _exact_one_sided_interval(
        successes, count, tail_alpha=tail_alpha
    )
    return {
        "success_count": successes,
        "replication_count": count,
        "estimate": successes / count,
        "simultaneous_lower": lower,
        "simultaneous_upper": upper,
        "one_sided_tail_alpha": tail_alpha,
    }


def _full_region_bounds(
    estimate: tf.Tensor, covariance: tf.Tensor
) -> tuple[tf.Tensor, ...]:
    bounds = predictive.batched_quadratic_loss_confidence_bounds(
        estimate, covariance, FULL_LOSS_MATRICES, alpha=0.05
    )
    return (
        bounds.point_loss[:, 0],
        bounds.lower_bound[:, 0],
        bounds.upper_bound[:, 0],
        bounds.point_loss[:, 1:],
        bounds.lower_bound[:, 1:],
        bounds.upper_bound[:, 1:],
        bounds.lower_kkt_residual[:, 0],
        bounds.upper_kkt_residual[:, 0],
        bounds.lower_kkt_residual[:, 1:],
        bounds.upper_kkt_residual[:, 1:],
        bounds.inference_admissible,
        bounds.confidence_radius_squared,
        tf.fill([HORIZON], bounds.confidence_radius_squared),
    )


def _split_region_bounds(
    estimate: tf.Tensor, covariance: tf.Tensor
) -> tuple[tf.Tensor, ...]:
    bounds = predictive.batched_split_quadratic_loss_confidence_bounds(
        estimate,
        covariance,
        AVERAGE_LOSS,
        average_alpha=AVERAGE_ALPHA,
        horizon_alpha=HORIZON_ALPHA,
        familywise_alpha=FAMILYWISE_REGION_ALPHA,
    )
    return (
        bounds.average_point_loss,
        bounds.average_lower_bound,
        bounds.average_upper_bound,
        bounds.horizon_point_losses,
        bounds.horizon_lower_bounds,
        bounds.horizon_upper_bounds,
        bounds.average_lower_kkt_residual,
        bounds.average_upper_kkt_residual,
        bounds.horizon_lower_kkt_residuals,
        bounds.horizon_upper_kkt_residuals,
        bounds.inference_admissible,
        bounds.average_confidence_radius_squared,
        tf.fill([HORIZON], bounds.horizon_confidence_radius_squared),
    )


def _family_batch(
    family: Family,
    *,
    family_index: int,
    draw_count: int,
    base_seed: tuple[int, int],
    batch_start: int,
    batch_size: int,
    geometry: Literal["full", "split"],
    estimator: Literal["path", "rao_blackwell"],
    hac_multiplier: float,
) -> dict[str, tf.Tensor]:
    root = tf.constant(base_seed, tf.int32)
    batch_seed = _fold(_fold(root, family_index), batch_start)
    components = _compiled("components", draw_count, batch_size)
    left = components(
        _fold(batch_seed, 11),
        tf.constant(family.phi, tf.float64),
        tf.constant(0.7, tf.float64),
    )
    right = components(
        _fold(batch_seed, 22),
        tf.constant(family.phi, tf.float64),
        tf.constant(family.horizon_rho, tf.float64),
    )
    right = _apply_family(*right, family)
    effective_estimator = (
        "path" if (family.skew or family.heavy_tail) else estimator
    )
    if effective_estimator == "path":
        feature = _compiled("path_features", draw_count, batch_size)
        left_estimate, left_influence = feature(left[0])
        right_estimate, right_influence = feature(right[0])
    else:
        feature = _compiled("conditional_features", draw_count, batch_size)
        left_estimate, left_influence = feature(left[1], left[2])
        right_estimate, right_influence = feature(right[1], right[2])
    estimate = left_estimate - right_estimate
    influence = tf.concat((2.0 * left_influence, -2.0 * right_influence), axis=1)
    hac = predictive.batched_chain_bartlett_long_run_covariance(
        influence,
        bandwidth_multiplier=hac_multiplier,
        ridge_ladder=RIDGE_LADDER,
        condition_number_max=CONDITION_NUMBER_MAX,
    )
    region = (
        _full_region_bounds(estimate, hac.regularized_covariance)
        if geometry == "full"
        else _split_region_bounds(estimate, hac.regularized_covariance)
    )
    (
        average_point,
        average_lower,
        average_upper,
        horizon_point,
        horizon_lower,
        horizon_upper,
        average_lower_kkt,
        average_upper_kkt,
        horizon_lower_kkt,
        horizon_upper_kkt,
        region_admissible,
        average_radius,
        horizon_radii,
    ) = region
    valid = tf.logical_and(hac.inference_admissible, region_admissible)
    passing = tf.logical_and(
        valid,
        tf.logical_and(
            average_upper < ACCEPTABLE_LOSS,
            tf.reduce_all(horizon_upper < ACCEPTABLE_LOSS, axis=1),
        ),
    )
    material = tf.logical_and(
        valid,
        tf.logical_or(
            average_lower > ACCEPTABLE_LOSS,
            tf.reduce_any(horizon_lower > ACCEPTABLE_LOSS, axis=1),
        ),
    )
    truth = _family_truth(family)
    centered = estimate - truth[None, :]
    average_quadratic = tf.einsum("bi,bij,bj->b", centered, hac.precision, centered)
    average_covered = tf.logical_and(valid, average_quadratic <= average_radius)
    horizon_indices = tf.stack(
        (tf.range(HORIZON, dtype=tf.int32), tf.range(HORIZON, dtype=tf.int32) + HORIZON),
        axis=1,
    )
    selectors = tf.one_hot(horizon_indices, FEATURE_COUNT, dtype=tf.float64)
    local_centered = tf.gather(centered, horizon_indices, axis=1)
    local_covariance = tf.einsum(
        "hki,bij,hlj->bhkl", selectors, hac.regularized_covariance, selectors
    )
    local_precision = tf.linalg.inv(local_covariance)
    horizon_quadratic = tf.einsum(
        "bhi,bhij,bhj->bh", local_centered, local_precision, local_centered
    )
    horizon_covered = tf.logical_and(
        valid[:, None], horizon_quadratic <= horizon_radii[None, :]
    )
    simultaneous_covered = tf.logical_and(
        average_covered, tf.reduce_all(horizon_covered, axis=1)
    )
    _require_gpu(
        estimate,
        hac.regularized_covariance,
        horizon_upper,
        surface="directional-region controlled remedy",
    )
    return {
        "estimate": estimate,
        "valid": valid,
        "pass": passing,
        "material": material,
        "average_covered": average_covered,
        "horizon_covered": horizon_covered,
        "simultaneous_covered": simultaneous_covered,
        "condition_number": hac.condition_number,
        "average_point": average_point,
        "average_lower": average_lower,
        "average_upper": average_upper,
        "horizon_point": horizon_point,
        "horizon_lower": horizon_lower,
        "horizon_upper": horizon_upper,
        "average_lower_kkt": average_lower_kkt,
        "average_upper_kkt": average_upper_kkt,
        "horizon_lower_kkt": horizon_lower_kkt,
        "horizon_upper_kkt": horizon_upper_kkt,
    }


def _summarize_family(
    family: Family,
    batches: list[dict[str, tf.Tensor]],
    replication_count: int,
    *,
    inferential: bool,
) -> dict[str, Any]:
    joined = {key: tf.concat([batch[key] for batch in batches], axis=0) for key in batches[0]}
    valid_count = int(tf.reduce_sum(tf.cast(joined["valid"], tf.int32)))
    invalid_count = replication_count - valid_count
    pass_count = int(tf.reduce_sum(tf.cast(joined["pass"], tf.int32)))
    material_count = int(tf.reduce_sum(tf.cast(joined["material"], tf.int32)))
    coverage_count = int(
        tf.reduce_sum(tf.cast(joined["simultaneous_covered"], tf.int32))
    )
    decisive_count = pass_count + material_count
    required_count = (
        pass_count if family.role == "equivalence" else material_count
        if family.role == "material" else 0
    )
    false_count = (
        material_count if family.role in {"equivalence", "guard_equivalence"}
        else pass_count if family.role in {"material", "guard_material"}
        else decisive_count
        if family.role == "boundary" else 0
    )
    tail_alpha = BINOMIAL_TAIL_ALPHA if inferential else 0.05
    coverage = _probability_record(coverage_count, replication_count, tail_alpha=tail_alpha)
    required = _probability_record(required_count, replication_count, tail_alpha=tail_alpha)
    false = _probability_record(false_count, replication_count, tail_alpha=tail_alpha)
    invalid = _probability_record(invalid_count, replication_count, tail_alpha=tail_alpha)
    if family.role in {"equivalence", "material"}:
        gate = {
            "coverage": coverage["simultaneous_lower"] >= MINIMUM_COVERAGE,
            "required_decision": required["simultaneous_lower"] >= MINIMUM_REQUIRED_DECISION,
            "false_decision": false["simultaneous_upper"] <= MAXIMUM_FALSE_DECISION,
            "invalid_procedure": invalid["simultaneous_upper"] <= MAXIMUM_INVALID,
        }
    elif family.role == "boundary":
        gate = {
            "coverage": coverage["simultaneous_lower"] >= MINIMUM_COVERAGE,
            "boundary_leakage": false["simultaneous_upper"] <= MAXIMUM_BOUNDARY_DECISION,
            "invalid_procedure": invalid["simultaneous_upper"] <= MAXIMUM_INVALID,
        }
    elif family.role in {"guard_equivalence", "guard_material"}:
        gate = {
            "coverage": coverage["simultaneous_lower"] >= MINIMUM_COVERAGE,
            "wrong_direction_decision": false["simultaneous_upper"] <= MAXIMUM_FALSE_DECISION,
            "invalid_procedure": invalid["simultaneous_upper"] <= MAXIMUM_INVALID,
        }
    else:
        gate = None
    if gate is not None:
        gate["passed"] = all(gate.values())
    return {
        "family": asdict(family),
        "truth": _family_truth(family).numpy().tolist(),
        "counts": {
            "pass": pass_count,
            "material_difference": material_count,
            "inconclusive": replication_count - invalid_count - decisive_count,
            "invalid": invalid_count,
        },
        "coverage": coverage,
        "required_decision": required,
        "false_or_boundary_decision": false,
        "invalid_procedure": invalid,
        "gate": gate,
        "descriptive": {
            "mean_average_point_loss": float(tf.reduce_mean(joined["average_point"])),
            "maximum_horizon_point_loss": float(tf.reduce_max(joined["horizon_point"])),
            "maximum_condition_number": float(tf.reduce_max(joined["condition_number"])),
            "mean_estimate": tf.reduce_mean(joined["estimate"], axis=0).numpy().tolist(),
        },
    }


def _run_candidate(
    *,
    name: str,
    geometry: Literal["full", "split"],
    estimator: Literal["path", "rao_blackwell"],
    hac_multiplier: float,
    families: tuple[Family, ...],
    draw_count: int,
    replication_count: int,
    batch_size: int,
    seed: tuple[int, int],
    deadline: float,
    inferential: bool,
) -> dict[str, Any]:
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    for family_index, family in enumerate(families):
        batches: list[dict[str, tf.Tensor]] = []
        for batch_start in range(0, replication_count, batch_size):
            if time.monotonic() >= deadline:
                raise RemedyError("wall cap exhausted during controlled remedy")
            current_batch = min(batch_size, replication_count - batch_start)
            batches.append(
                _family_batch(
                    family,
                    family_index=family_index,
                    draw_count=draw_count,
                    base_seed=seed,
                    batch_start=batch_start,
                    batch_size=current_batch,
                    geometry=geometry,
                    estimator=estimator,
                    hac_multiplier=hac_multiplier,
                )
            )
        rows.append(
            _summarize_family(
                family, batches, replication_count, inferential=inferential
            )
        )
    gated = [row for row in rows if row["gate"] is not None]
    return {
        "name": name,
        "geometry": geometry,
        "estimator": estimator,
        "shape_stress_estimator_override": (
            "path for skew/heavy-tail explanatory families"
        ),
        "hac_multiplier": hac_multiplier,
        "draw_count_per_chain": draw_count,
        "replication_count_per_family": replication_count,
        "batch_size": batch_size,
        "seed": list(seed),
        "bandwidth": predictive.growing_hac_bandwidth(
            draw_count, multiplier=hac_multiplier
        ),
        "families": rows,
        "gate_passed": bool(gated) and all(row["gate"]["passed"] for row in gated),
        "wall_seconds": time.monotonic() - started,
    }


def _development_candidates() -> tuple[tuple[str, str, str, float], ...]:
    return (
        ("baseline_full_path_k1", "full", "path", 1.0),
        ("split_path_k1", "split", "path", 1.0),
        *((f"split_path_k{str(k).replace('.', 'p')}", "split", "path", k) for k in HAC_CANDIDATES[1:]),
        *((f"split_rao_k{str(k).replace('.', 'p')}", "split", "rao_blackwell", k) for k in HAC_CANDIDATES),
    )


def _nominate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    split = [candidate for candidate in candidates if candidate["geometry"] == "split"]
    if not split:
        raise RemedyError("development produced no split-region candidate")

    def score(candidate: dict[str, Any]) -> tuple[float, float, float, float, float]:
        safety = [
            row for row in candidate["families"]
            if row["family"]["role"] in {
                "equivalence", "material", "guard_equivalence", "guard_material", "boundary"
            }
        ]
        primary = [
            row for row in safety
            if row["family"]["role"] in {"equivalence", "material"}
        ]
        valid = min(1.0 - row["invalid_procedure"]["estimate"] for row in safety)
        coverage = min(row["coverage"]["estimate"] for row in safety)
        false = max(row["false_or_boundary_decision"]["estimate"] for row in safety)
        required = min(row["required_decision"]["estimate"] for row in primary)
        rao = 1.0 if candidate["estimator"] == "rao_blackwell" else 0.0
        return valid, coverage, -false, required, rao

    selected = max(split, key=score)
    return {
        "candidate_name": selected["name"],
        "geometry": selected["geometry"],
        "estimator": selected["estimator"],
        "hac_multiplier": selected["hac_multiplier"],
        "selection_rule": (
            "lexicographic descriptive nomination: minimum validity, minimum "
            "coverage, negative maximum false rate, minimum required-decision "
            "rate, then Rao-Blackwell tie preference"
        ),
        "statistical_ranking_supported": False,
    }


def _validate_development_receipt(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _strict_json(path)
    if (
        payload.get("schema") != "bayesfilter.ssl_lstm.directional_region_remedy.v1"
        or payload.get("configuration", {}).get("mode") != "development"
        or payload.get("claim_boundary", {}).get("statistical_promotion_evidence") is not False
    ):
        raise RemedyError("development receipt contract drift")
    nomination = payload.get("nomination")
    if not isinstance(nomination, dict) or nomination.get("geometry") != "split":
        raise RemedyError("development receipt lacks a valid split nomination")
    return payload, nomination


def _capacity_qualifies(candidate: dict[str, Any]) -> bool:
    gated = [row for row in candidate["families"] if row["gate"] is not None]
    primary = [
        row for row in gated
        if row["family"]["role"] in {"equivalence", "material"}
    ]
    covered = sum(row["coverage"]["success_count"] for row in gated)
    total = sum(row["coverage"]["replication_count"] for row in gated)
    pooled_coverage = covered / total
    return (
        bool(gated)
        and CAPACITY_MINIMUM_POOLED_COVERAGE
        <= pooled_coverage
        <= CAPACITY_MAXIMUM_POOLED_COVERAGE
        and min(row["required_decision"]["estimate"] for row in primary)
        >= CAPACITY_MINIMUM_REQUIRED_DECISION
        and max(row["false_or_boundary_decision"]["estimate"] for row in gated)
        <= CAPACITY_MAXIMUM_FALSE_DECISION
        and max(row["invalid_procedure"]["estimate"] for row in gated) == 0.0
    )


def _validate_capacity_receipt(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _strict_json(path)
    if (
        payload.get("schema") != "bayesfilter.ssl_lstm.directional_region_remedy.v1"
        or payload.get("configuration", {}).get("mode") != "capacity"
        or payload.get("claim_boundary", {}).get("statistical_promotion_evidence") is not False
    ):
        raise RemedyError("capacity receipt contract drift")
    nomination = payload.get("audit_nomination")
    if not isinstance(nomination, dict) or nomination.get("audit_authorized") is not True:
        raise RemedyError("capacity receipt does not authorize a locked audit")
    return payload, nomination


def _nominate_capacity_from_receipt(path: Path) -> dict[str, Any]:
    payload = _strict_json(path)
    if (
        payload.get("schema") != "bayesfilter.ssl_lstm.directional_region_remedy.v1"
        or payload.get("configuration", {}).get("mode") != "capacity"
        or payload.get("claim_boundary", {}).get("statistical_promotion_evidence") is not False
    ):
        raise RemedyError("capacity receipt contract drift")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise RemedyError("capacity receipt lacks candidate rows")
    qualified = next(
        (candidate for candidate in candidates if _capacity_qualifies(candidate)),
        None,
    )
    if qualified is None:
        raise RemedyError("capacity receipt has no audit-capable rung")
    return {
        "audit_authorized": True,
        "draw_count": qualified["draw_count_per_chain"],
        "geometry": qualified["geometry"],
        "estimator": qualified["estimator"],
        "hac_multiplier": qualified["hac_multiplier"],
        "capacity_receipt_sha256": _sha256(path),
        "statistical_promotion_evidence": False,
        "repair_note": "minimum-across-96 coverage gate removed after exact false-stop audit",
    }


def run(
    *,
    mode: str,
    output: Path,
    wall_cap_seconds: float,
    development_receipt: Path | None = None,
) -> dict[str, Any]:
    if mode not in {"smoke", "development", "capacity", "audit"}:
        raise RemedyError("HMC, NeuTra training, and confirmation remain closed")
    if not math.isfinite(wall_cap_seconds) or wall_cap_seconds <= 0.0:
        raise RemedyError("wall_cap_seconds must be finite and positive")
    if _absolute(output).exists():
        raise RemedyError(f"refusing to overwrite receipt: {output}")
    if not tf.config.list_physical_devices("GPU"):
        raise RemedyError("trusted GPU is required")
    started = time.monotonic()
    deadline = started + wall_cap_seconds
    started_at = _now()
    candidates: list[dict[str, Any]] = []
    nomination = None
    development_binding = None
    capacity_binding = None
    if mode == "smoke":
        candidates.append(
            _run_candidate(
                name="split_rao_smoke",
                geometry="split",
                estimator="rao_blackwell",
                hac_multiplier=1.0,
                families=(FAMILIES[1], FAMILIES[6], FAMILIES[11]),
                draw_count=SMOKE_DRAW_COUNT,
                replication_count=SMOKE_REPLICATION_COUNT,
                batch_size=SMOKE_REPLICATION_COUNT,
                seed=SMOKE_SEED,
                deadline=deadline,
                inferential=False,
            )
        )
    elif mode == "development":
        for name, geometry, estimator, multiplier in _development_candidates():
            candidates.append(
                _run_candidate(
                    name=name,
                    geometry=geometry,  # type: ignore[arg-type]
                    estimator=estimator,  # type: ignore[arg-type]
                    hac_multiplier=multiplier,
                    families=PRIMARY_FAMILIES + BOUNDARY_FAMILIES + GUARD_FAMILIES,
                    draw_count=DRAW_COUNT,
                    replication_count=DEVELOPMENT_REPLICATION_COUNT,
                    batch_size=DEVELOPMENT_BATCH_SIZE,
                    seed=DEVELOPMENT_SEED,
                    deadline=deadline,
                    inferential=False,
                )
            )
        nomination = _nominate(candidates)
    elif mode == "capacity":
        receipt_path = DEVELOPMENT_RECEIPT if development_receipt is None else development_receipt
        _, nomination = _validate_development_receipt(receipt_path)
        development_binding = {
            "path": str(receipt_path),
            "sha256": _sha256(receipt_path),
        }
        for rung_index, draw_count in enumerate(CAPACITY_DRAW_LADDER):
            candidate = _run_candidate(
                name=f"capacity_{draw_count}",
                geometry=nomination["geometry"],
                estimator=nomination["estimator"],
                hac_multiplier=float(nomination["hac_multiplier"]),
                families=PRIMARY_FAMILIES + BOUNDARY_FAMILIES + GUARD_FAMILIES,
                draw_count=draw_count,
                replication_count=CAPACITY_REPLICATION_COUNT,
                batch_size=DEVELOPMENT_BATCH_SIZE,
                seed=(DEVELOPMENT_SEED[0] + 100 + rung_index, DEVELOPMENT_SEED[1]),
                deadline=deadline,
                inferential=False,
            )
            candidate["capacity_qualified"] = _capacity_qualifies(candidate)
            candidates.append(candidate)
            if candidate["capacity_qualified"]:
                break
        qualified = next(
            (candidate for candidate in candidates if candidate["capacity_qualified"]),
            None,
        )
        audit_nomination = {
            "audit_authorized": qualified is not None,
            "draw_count": None if qualified is None else qualified["draw_count_per_chain"],
            "geometry": nomination["geometry"],
            "estimator": nomination["estimator"],
            "hac_multiplier": nomination["hac_multiplier"],
            "development_receipt_sha256": development_binding["sha256"],
            "statistical_promotion_evidence": False,
        }
    else:
        receipt_path = CAPACITY_RECEIPT if development_receipt is None else development_receipt
        capacity_payload = _strict_json(receipt_path)
        audit_nomination = _nominate_capacity_from_receipt(receipt_path)
        capacity_binding = {
            "path": str(receipt_path),
            "sha256": _sha256(receipt_path),
            "original_decision": capacity_payload.get("decision"),
            "original_audit_authorized": capacity_payload.get(
                "audit_nomination", {}
            ).get("audit_authorized"),
            "post_run_review_repair": audit_nomination["repair_note"],
        }
        candidates.append(
            _run_candidate(
                name="locked_audit_candidate",
                geometry=audit_nomination["geometry"],
                estimator=audit_nomination["estimator"],
                hac_multiplier=float(audit_nomination["hac_multiplier"]),
                families=FAMILIES,
                draw_count=int(audit_nomination["draw_count"]),
                replication_count=AUDIT_REPLICATION_COUNT,
                batch_size=AUDIT_BATCH_SIZE,
                seed=AUDIT_SEED,
                deadline=deadline,
                inferential=True,
            )
        )
    if mode == "smoke":
        decision = "DIRECTIONAL_REGION_SMOKE_PASSED_DEVELOPMENT_REQUIRED"
    elif mode == "development":
        decision = "DEVELOPMENT_NOMINATION_COMPLETE_NOT_PROMOTION_EVIDENCE"
    elif mode == "capacity":
        decision = (
            "CAPACITY_NOMINATION_COMPLETE_AUDIT_AUTHORIZED"
            if audit_nomination["audit_authorized"]
            else "CAPACITY_LADDER_FAILED_AUDIT_REMAINS_CLOSED"
        )
    elif candidates[0]["gate_passed"]:
        decision = "LOCKED_CONTROLLED_AUDIT_PASSED_TARGET_CONFIRMATION_STILL_CLOSED"
    else:
        decision = "LOCKED_CONTROLLED_AUDIT_CANDIDATE_REJECTED_REPAIR_REQUIRED"
    payload = {
        "schema": "bayesfilter.ssl_lstm.directional_region_remedy.v1",
        "decision": decision,
        "started_at_utc": started_at,
        "finished_at_utc": _now(),
        "configuration": {
            "mode": mode,
            "chain_count": CHAIN_COUNT,
            "forecast_replication_count": FORECAST_REPLICATION_COUNT,
            "horizon": HORIZON,
            "draw_count": DRAW_COUNT,
            "capacity_draw_ladder": list(CAPACITY_DRAW_LADDER),
            "development_replication_count": DEVELOPMENT_REPLICATION_COUNT,
            "audit_replication_count": AUDIT_REPLICATION_COUNT,
            "average_alpha": AVERAGE_ALPHA,
            "horizon_alpha": HORIZON_ALPHA,
            "region_familywise_alpha": FAMILYWISE_REGION_ALPHA,
            "hac_candidates": list(HAC_CANDIDATES),
            "ridge_ladder": list(RIDGE_LADDER),
            "scale_source": "controlled_law_known_unit_scale",
            "scale_uses_confirmation_outcomes": False,
            "acceptable_loss": ACCEPTABLE_LOSS,
            "audit_operating_claim_count": AUDIT_OPERATING_CLAIMS,
            "binomial_tail_alpha": BINOMIAL_TAIL_ALPHA,
            "capacity_minimum_pooled_coverage": CAPACITY_MINIMUM_POOLED_COVERAGE,
            "capacity_maximum_pooled_coverage": CAPACITY_MAXIMUM_POOLED_COVERAGE,
            "capacity_minimum_required_decision": CAPACITY_MINIMUM_REQUIRED_DECISION,
            "capacity_maximum_false_decision": CAPACITY_MAXIMUM_FALSE_DECISION,
            "boundary_mean": BOUNDARY_MEAN,
            "boundary_variance_up": BOUNDARY_VARIANCE_UP,
            "boundary_variance_down": BOUNDARY_VARIANCE_DOWN,
        },
        "development_binding": development_binding,
        "capacity_binding": capacity_binding,
        "nomination": nomination,
        "audit_nomination": audit_nomination if mode in {"capacity", "audit"} else None,
        "candidates": candidates,
        "claim_boundary": {
            "statistical_promotion_evidence": mode == "audit",
            "development_is_nomination_only": mode != "audit",
            "controlled_laws_only": True,
            "hmc_authorized": False,
            "neutra_training_authorized": False,
            "confirmation_forecast_bank_opened": False,
            "g_h_predictive_difference_computed": False,
            "posterior_correctness_concluded": False,
            "statistical_ranking_supported": False,
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
            "logical_devices": [device.name for device in tf.config.list_logical_devices()],
            "float_type": "float64",
            "jit_compile": True,
            "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
            "wall_cap_seconds": wall_cap_seconds,
            "wall_seconds": time.monotonic() - started,
            "plan_path": str(PLAN_PATH),
            "output_path": str(output),
            "source_hashes": {
                str(PLAN_PATH): _sha256(PLAN_PATH),
                str(SCRIPT_PATH): _sha256(SCRIPT_PATH),
                str(PREDICTIVE_SOURCE): _sha256(PREDICTIVE_SOURCE),
            },
        },
    }
    _write(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("smoke", "development", "capacity", "audit"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-cap-seconds", type=float, required=True)
    parser.add_argument("--development-receipt", type=Path)
    arguments = parser.parse_args()
    payload = run(
        mode=arguments.mode,
        output=arguments.output,
        wall_cap_seconds=arguments.wall_cap_seconds,
        development_receipt=arguments.development_receipt,
    )
    print(json.dumps({"decision": payload["decision"], "output": str(arguments.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
