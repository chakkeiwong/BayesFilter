"""Structured one- and two-factor covariance fits for local score geometry.

For standardized offsets ``z`` around an exact center ``c``, the fitter uses

``g(c) - g(c + z) ~= P z``, where ``P = C^{-1}`` and

``C = D [diag(1 - row_norm(L)^2) + L L^T] D``.

The row-ball loading transform keeps the correlation matrix strictly SPD.
Factor loadings are nuisance coordinates: the implied covariance, precision,
and held-out score prediction are the numerical objects consumed downstream.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.compiler.tf2xla.ops.gen_xla_ops import xla_svd

from bayesfilter.inference.mass_matrix_tf import _eigenpairs
from bayesfilter.ops.host_tensor_io import numeric_tensor
from bayesfilter.ops.qr_lstsq_tf import complete_orthogonal_lstsq

FACTOR_CORRELATION_GEOMETRY_NONCLAIMS = (
    "structured local score geometry diagnostic only",
    "not a certified MAP",
    "not a certified posterior covariance",
    "not HMC readiness evidence",
    "not convergence evidence",
    "not default-readiness evidence",
)


@dataclass(frozen=True)
class FactorCorrelationGeometryConfig:
    """Numerical policy for one structured covariance fit."""

    factor_count: int = 1
    loading_margin: float = 1.0e-6
    standard_deviation_floor: float = 1.0e-6
    max_condition_number: float = 1.0e8
    holdout_score_relative_rmse: float = 0.35
    max_iterations: int = 200
    tolerance: float = 1.0e-9

    def __post_init__(self) -> None:
        factors = int(self.factor_count)
        if factors not in (1, 2):
            raise ValueError("factor_count must be one or two")
        object.__setattr__(self, "factor_count", factors)
        for name in (
            "loading_margin",
            "standard_deviation_floor",
            "holdout_score_relative_rmse",
            "tolerance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive finite")
            object.__setattr__(self, name, value)
        if self.loading_margin >= 1.0:
            raise ValueError("loading_margin must be less than one")
        condition = float(self.max_condition_number)
        if not math.isfinite(condition) or condition <= 1.0:
            raise ValueError("max_condition_number must exceed one")
        object.__setattr__(self, "max_condition_number", condition)
        iterations = int(self.max_iterations)
        if iterations <= 0:
            raise ValueError("max_iterations must be positive")
        object.__setattr__(self, "max_iterations", iterations)


@dataclass(frozen=True)
class FactorCorrelationGeometryResult:
    """Result of one structured local score-geometry fit."""

    accepted: bool
    status: str
    factor_count: int
    parameter_count: int
    anchor_indices: tuple[int, ...]
    covariance_z: tf.Tensor | None
    precision_z: tf.Tensor | None
    marginal_standard_deviations: tf.Tensor | None
    loadings: tf.Tensor | None
    diagnostics: Mapping[str, Any]
    nonclaims: tuple[str, ...] = FACTOR_CORRELATION_GEOMETRY_NONCLAIMS

    def __post_init__(self) -> None:
        object.__setattr__(self, "accepted", bool(self.accepted))
        object.__setattr__(self, "status", str(self.status))
        object.__setattr__(self, "factor_count", int(self.factor_count))
        object.__setattr__(self, "parameter_count", int(self.parameter_count))
        object.__setattr__(
            self, "anchor_indices", tuple(int(value) for value in self.anchor_indices)
        )
        for name in (
            "covariance_z",
            "precision_z",
            "marginal_standard_deviations",
            "loadings",
        ):
            value = getattr(self, name)
            if value is not None:
                array = tf.identity(numeric_tensor(value, tf.float64))
                object.__setattr__(self, name, array)
        object.__setattr__(self, "diagnostics", _json_ready(dict(self.diagnostics)))

    def payload(self) -> Mapping[str, Any]:
        return _json_ready(
            {
                "schema": "bayesfilter.factor_correlation_geometry.v1",
                "accepted": self.accepted,
                "status": self.status,
                "factor_count": self.factor_count,
                "parameter_count": self.parameter_count,
                "anchor_indices": self.anchor_indices,
                "covariance_z": self.covariance_z,
                "precision_z": self.precision_z,
                "marginal_standard_deviations": self.marginal_standard_deviations,
                "loadings": self.loadings,
                "diagnostics": self.diagnostics,
                "nonclaims": self.nonclaims,
            }
        )


def fit_factor_correlation_score_geometry(
    center_score_z: Any,
    training_offsets_z: Any,
    training_scores_z: Any,
    holdout_offsets_z: Any,
    holdout_scores_z: Any,
    *,
    training_weights: Any | None = None,
    config: FactorCorrelationGeometryConfig | None = None,
    jit_compile: bool = True,
) -> FactorCorrelationGeometryResult:
    """Fit an SPD factor covariance to exact local score differences.

    The center score and all scores must already be expressed in the same
    standardized ``z`` coordinates. Holdout rows are never used to initialize
    or optimize the fit.
    """

    cfg = FactorCorrelationGeometryConfig() if config is None else config
    center = tf.reshape(tf.convert_to_tensor(center_score_z, tf.float64), [-1])
    train_z = tf.convert_to_tensor(training_offsets_z, tf.float64)
    train_score = tf.convert_to_tensor(training_scores_z, tf.float64)
    holdout_z = tf.convert_to_tensor(holdout_offsets_z, tf.float64)
    holdout_score = tf.convert_to_tensor(holdout_scores_z, tf.float64)
    if center.shape[0] is None:
        raise ValueError("center_score_z must have static dimension")
    dimension = int(center.shape[0])
    nominal_parameter_count = (
        2 * dimension if cfg.factor_count == 1 else 3 * dimension - 1
    )
    symmetric_entry_count = dimension * (dimension + 1) // 2
    if nominal_parameter_count > symmetric_entry_count:
        return _rejected(
            cfg,
            dimension,
            "factor_parameterization_dimensionally_unidentified",
            parameter_count=nominal_parameter_count,
            diagnostics={
                "parameter_count": nominal_parameter_count,
                "symmetric_covariance_entry_count": symmetric_entry_count,
            },
        )
    _require_matrix(train_z, "training_offsets_z", dimension)
    _require_matrix(train_score, "training_scores_z", dimension)
    _require_matrix(holdout_z, "holdout_offsets_z", dimension)
    _require_matrix(holdout_score, "holdout_scores_z", dimension)
    if train_z.shape != train_score.shape or holdout_z.shape != holdout_score.shape:
        raise ValueError("offset and score matrices must have matching shapes")
    if int(train_z.shape[0]) <= 0 or int(holdout_z.shape[0]) <= 0:
        raise ValueError("training and holdout rows must both be nonempty")
    all_finite = tf.reduce_all(
        tf.math.is_finite(center)
        & tf.reduce_all(tf.math.is_finite(train_z), axis=0)
        & tf.reduce_all(tf.math.is_finite(train_score), axis=0)
        & tf.reduce_all(tf.math.is_finite(holdout_z), axis=0)
        & tf.reduce_all(tf.math.is_finite(holdout_score), axis=0)
    )
    if not bool(all_finite.numpy()):
        return _rejected(cfg, dimension, "nonfinite_fit_inputs")

    row_count = int(train_z.shape[0])
    weights = (
        tf.fill([row_count], tf.constant(1.0 / row_count, tf.float64))
        if training_weights is None
        else tf.reshape(tf.convert_to_tensor(training_weights, tf.float64), [-1])
    )
    if weights.shape != (row_count,) or not bool(
        tf.reduce_all(tf.math.is_finite(weights) & (weights > 0.0)).numpy()
    ):
        raise ValueError("training_weights must be positive finite per row")
    parameter_count = nominal_parameter_count
    program = _make_factor_program(dimension, row_count, int(holdout_z.shape[0]), cfg,
                                   bool(jit_compile), _prediction_jacobian_diagnostics)
    try:
        computed = program(center, train_z, train_score, holdout_z, holdout_score, weights)
    except (tf.errors.OpError, ValueError) as exc:
        return _rejected(cfg, dimension, "factor_optimizer_failed", parameter_count=parameter_count,
                         diagnostics={"exception_type": type(exc).__name__, "jit_compile": bool(jit_compile)})
    covariance = computed["covariance"]
    precision = computed["precision"]
    deviations = computed["deviations"]
    loadings = computed["loadings"]
    eigenvalues = computed["eigenvalues"]
    finite = computed["finite"]
    condition_number = computed["condition_number"]
    train_rmse = computed["train_rmse"]
    holdout_error = computed["holdout_error"]
    holdout_relative = computed["holdout_relative"]
    jacobian_rank = computed["jacobian_rank"]
    jacobian_condition = computed["jacobian_condition"]
    optimizer = computed["optimizer"]
    anchors = tuple(computed["anchors"].numpy().tolist())
    jacobian_rank = int(jacobian_rank)
    jacobian_condition = None if jacobian_rank == 0 else float(jacobian_condition)
    second_factor_identified = bool(
        cfg.factor_count == 1 or jacobian_rank == parameter_count
    )
    status = "usable"
    if not bool(finite.numpy()) or not bool(tf.reduce_min(eigenvalues).numpy() > 0.0):
        status = "nonfinite_or_non_spd_fit"
    elif float(condition_number.numpy()) > cfg.max_condition_number * (1.0 + 1.0e-8):
        status = "condition_number_above_cap"
    elif cfg.factor_count == 2 and not second_factor_identified:
        status = "second_factor_unidentified"
    elif float(holdout_relative.numpy()) > cfg.holdout_score_relative_rmse:
        status = "holdout_score_fit_rejected"
    elif bool(optimizer.failed.numpy()):
        status = "factor_optimizer_failed"

    diagnostics = {
        "jit_compile": bool(jit_compile),
        "training_row_count": row_count,
        "holdout_row_count": int(holdout_z.shape[0]),
        "training_score_equation_count": row_count * dimension,
        "holdout_score_equation_count": int(holdout_z.shape[0]) * dimension,
        "parameter_count": parameter_count,
        "train_score_rmse": float(train_rmse.numpy()),
        "holdout_score_rmse": float(holdout_error.numpy()),
        "holdout_score_relative_rmse": float(holdout_relative.numpy()),
        "holdout_score_relative_rmse_cap": cfg.holdout_score_relative_rmse,
        "covariance_eigenvalues": eigenvalues.numpy(),
        "condition_number": float(condition_number.numpy()),
        "prediction_jacobian_rank": jacobian_rank,
        "prediction_jacobian_condition_number": jacobian_condition,
        "second_factor_identified": second_factor_identified,
        "optimizer_converged": bool(optimizer.converged.numpy()),
        "optimizer_failed": bool(optimizer.failed.numpy()),
        "optimizer_iterations": int(optimizer.num_iterations.numpy()),
        "optimizer_objective_evaluations": int(
            optimizer.num_objective_evaluations.numpy()
        ),
        "final_loss": float(optimizer.objective_value.numpy()),
        "loading_row_squared_norms": tf.reduce_sum(
            tf.square(loadings), axis=1
        ).numpy(),
        "covariance_parameterization": (
            "D[diag(1-row_norm(L)^2)+LL^T]D"
        ),
        "score_model": "center_score_minus_local_score_equals_precision_times_offset",
    }
    return FactorCorrelationGeometryResult(
        accepted=status == "usable",
        status=status,
        factor_count=cfg.factor_count,
        parameter_count=parameter_count,
        anchor_indices=anchors,
        covariance_z=covariance.numpy(),
        precision_z=precision.numpy(),
        marginal_standard_deviations=deviations.numpy(),
        loadings=loadings.numpy(),
        diagnostics=diagnostics,
    )


@lru_cache(maxsize=16)
def _make_factor_program(dimension, training_rows, holdout_rows, cfg, jit_compile, diagnosis,
                         *, padded_training=False):
    """One stable numerical boundary, including preparation and optimizer."""
    signature = [tf.TensorSpec([dimension], tf.float64),
        tf.TensorSpec([training_rows, dimension], tf.float64),
        tf.TensorSpec([training_rows, dimension], tf.float64),
        tf.TensorSpec([holdout_rows, dimension], tf.float64),
        tf.TensorSpec([holdout_rows, dimension], tf.float64),
        tf.TensorSpec([training_rows], tf.float64)]
    if padded_training:
        if training_rows < 2 * dimension:
            raise ValueError("padded structured training requires capacity for at least 2D fresh rows")
        signature.append(tf.TensorSpec([], tf.int32))
    @tf.function(input_signature=signature, jit_compile=jit_compile, autograph=False)
    def numerical(center, train_z, train_score, holdout_z, holdout_score, weights,
                  active_training_rows=None):
        if padded_training:
            active = tf.range(training_rows) < active_training_rows
            train_z = tf.where(active[:, None], train_z, tf.zeros_like(train_z))
            train_score = tf.where(active[:, None], train_score, center[None, :])
            weights = tf.where(active, weights, tf.zeros_like(weights))
        weights /= tf.reduce_sum(weights)
        train_response = center[None, :] - train_score
        holdout_response = center[None, :] - holdout_score

        dense_precision = _weighted_dense_precision(
            train_z,
            train_response,
            weights,
            max_condition_number=cfg.max_condition_number,
            jit_compile=jit_compile,
        )
        dense_covariance = tf.linalg.inv(dense_precision)
        initial_standard_deviations, initial_loadings, anchors = _initial_factor_state(
            dense_covariance,
            factor_count=cfg.factor_count,
            loading_margin=cfg.loading_margin,
            jit_compile=jit_compile,
        )
        initial_raw = _encode_state(
            initial_standard_deviations,
            initial_loadings,
            anchors,
            cfg,
        )
        parameter_count = int(initial_raw.shape[0])

        def loss(raw: tf.Tensor) -> tf.Tensor:
            covariance, _std, _loads = _decode_covariance(
                raw, dimension=dimension, anchors=anchors, config=cfg
            )
            precision = tf.linalg.cholesky_solve(
                tf.linalg.cholesky(covariance), tf.eye(dimension, dtype=tf.float64)
            )
            prediction = tf.einsum("ij,bj->bi", precision, train_z)
            per_row = tf.reduce_mean(tf.square(prediction - train_response), axis=1)
            return tf.reduce_sum(weights * per_row)

        @tf.function(input_signature=[tf.TensorSpec([parameter_count], tf.float64)],
            jit_compile=jit_compile, autograph=False)
        def value_and_gradient(raw: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            return tfp.math.value_and_gradient(loss, raw)

        optimizer = tfp.optimizer.lbfgs_minimize(
            value_and_gradient,
            initial_position=initial_raw,
            tolerance=tf.constant(cfg.tolerance, tf.float64),
            max_iterations=cfg.max_iterations,
            parallel_iterations=1,
        )
        fitted_raw = tf.convert_to_tensor(optimizer.position, tf.float64)
        covariance, deviations, loadings = _decode_covariance(
            fitted_raw, dimension=dimension, anchors=anchors, config=cfg
        )
        chol = tf.linalg.cholesky(covariance)
        precision = tf.linalg.cholesky_solve(
            chol, tf.eye(dimension, dtype=tf.float64)
        )

        train_prediction = tf.einsum("ij,bj->bi", precision, train_z)
        holdout_prediction = tf.einsum("ij,bj->bi", precision, holdout_z)
        train_rmse = tf.sqrt(
            tf.reduce_sum(
                weights
                * tf.reduce_mean(tf.square(train_prediction - train_response), axis=1)
            )
        )
        holdout_error = tf.sqrt(
            tf.reduce_mean(tf.square(holdout_prediction - holdout_response))
        )
        holdout_scale = tf.maximum(
            tf.sqrt(tf.reduce_mean(tf.square(holdout_response))),
            tf.constant(1.0e-15, tf.float64),
        )
        holdout_relative = holdout_error / holdout_scale
        eigenvalues = tf.linalg.eigvalsh(covariance)
        condition_number = tf.reduce_max(eigenvalues) / tf.reduce_min(eigenvalues)
        finite = (
            tf.reduce_all(tf.math.is_finite(covariance))
            & tf.reduce_all(tf.math.is_finite(precision))
            & tf.reduce_all(tf.math.is_finite(deviations))
            & tf.reduce_all(tf.math.is_finite(loadings))
        )
        diagnosis_arguments = {"dimension": dimension, "anchors": anchors, "config": cfg,
                               "jit_compile": jit_compile}
        if padded_training:
            finite &= (active_training_rows >= 2 * dimension) & (active_training_rows <= training_rows)
            jacobian_rank, jacobian_condition = diagnosis(fitted_raw, train_z,
                active_training_rows=active_training_rows, **diagnosis_arguments)
        else:
            jacobian_rank, jacobian_condition = diagnosis(fitted_raw, train_z, **diagnosis_arguments)
        # The original fit returned frozen host records. Keep its completed
        # geometry disconnected from an external tape without changing the
        # loss/gradient used by L-BFGS or the covariance constructor API.
        return tf.nest.map_structure(tf.stop_gradient,
            {"covariance": covariance, "precision": precision, "deviations": deviations,
             "loadings": loadings, "eigenvalues": eigenvalues, "finite": finite,
             "condition_number": condition_number, "train_rmse": train_rmse,
             "holdout_error": holdout_error, "holdout_relative": holdout_relative,
             "jacobian_rank": jacobian_rank, "jacobian_condition": jacobian_condition,
             "optimizer": optimizer, "anchors": tf.stack(anchors)})
    return numerical


def factor_correlation_covariance(
    marginal_standard_deviations: Any,
    loadings: Any,
    *,
    loading_margin: float = 1.0e-6,
) -> tf.Tensor:
    """Construct ``D R D`` and enforce the row-ball SPD contract."""

    deviations = tf.reshape(
        tf.convert_to_tensor(marginal_standard_deviations, tf.float64), [-1]
    )
    factors = tf.convert_to_tensor(loadings, tf.float64)
    if factors.shape.rank != 2 or factors.shape[0] != deviations.shape[0]:
        raise ValueError("loadings must have one row per marginal deviation")
    margin = tf.convert_to_tensor(loading_margin, tf.float64)
    row_norm_squared = tf.reduce_sum(tf.square(factors), axis=1)
    tf.debugging.assert_positive(deviations)
    tf.debugging.assert_less(row_norm_squared, 1.0 - margin)
    correlation = tf.linalg.diag(1.0 - row_norm_squared) + tf.matmul(
        factors, factors, transpose_b=True
    )
    return deviations[:, None] * correlation * deviations[None, :]


def _decode_covariance(
    raw: tf.Tensor,
    *,
    dimension: int,
    anchors: tuple[int, ...],
    config: FactorCorrelationGeometryConfig,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    vector = tf.reshape(tf.convert_to_tensor(raw, tf.float64), [-1])
    deviations = config.standard_deviation_floor + tf.nn.softplus(
        vector[:dimension]
    )
    bound = tf.sqrt(tf.constant(1.0 - config.loading_margin, tf.float64))
    remaining = vector[dimension:]
    if config.factor_count == 1:
        anchor = anchors[0]
        loadings = bound * tf.where(tf.range(dimension) == anchor,
            tf.math.sigmoid(remaining), tf.math.tanh(remaining))[:, None]
    else:
        anchor_a, anchor_b = anchors
        # Insert the omitted second coordinate of the first anchor. Every row
        # then has the same two-coordinate representation for tensor algebra.
        omitted = 2 * anchor_a + 1
        indices = tf.range(2*dimension)
        full = tf.where(indices == omitted, tf.zeros([2*dimension], tf.float64),
            tf.gather(remaining, tf.minimum(indices-tf.cast(indices > omitted, tf.int32), 2*dimension-2)))
        unconstrained = tf.reshape(full, [dimension, 2])
        loadings = bound * unconstrained / tf.sqrt(1.0 + tf.reduce_sum(tf.square(unconstrained), axis=1, keepdims=True))
        anchor_rows = tf.gather(unconstrained, tf.stack((anchor_a, anchor_b)))
        first_radius = bound * tf.math.sigmoid(anchor_rows[0, 0])
        second_radius = bound * tf.math.sigmoid(anchor_rows[1, 0])
        angle = tf.constant(math.pi, tf.float64) * tf.math.sigmoid(anchor_rows[1, 1])
        loadings = tf.tensor_scatter_nd_update(loadings, tf.reshape(tf.stack((anchor_a, anchor_b)), [2, 1]),
            tf.stack((tf.stack((first_radius, tf.constant(0., tf.float64))),
                      second_radius * tf.stack((tf.cos(angle), tf.sin(angle))))))
    covariance = factor_correlation_covariance(
        deviations, loadings, loading_margin=config.loading_margin
    )
    return covariance, deviations, loadings


def _initial_factor_state(
    covariance: tf.Tensor,
    *,
    factor_count: int,
    loading_margin: float,
    jit_compile: bool = True,
) -> tuple[tf.Tensor, tf.Tensor, tuple[int, ...]]:
    deviations = tf.sqrt(tf.linalg.diag_part(covariance))
    correlation = covariance / (deviations[:, None] * deviations[None, :])
    eigenvalues, eigenvectors = _eigenpairs(correlation, jit_compile)
    selected_values = eigenvalues[-factor_count:]
    selected_vectors = eigenvectors[:, -factor_count:]
    excess = tf.sqrt(tf.maximum(selected_values - 1.0, 1.0e-6))
    loadings = selected_vectors * excess[None, :]
    bound = math.sqrt(1.0 - loading_margin)
    row_norms = tf.linalg.norm(loadings, axis=1, keepdims=True)
    loadings *= tf.minimum(
        tf.ones_like(row_norms),
        tf.constant(0.8 * bound, tf.float64) / tf.maximum(row_norms, 1.0e-15),
    )
    if factor_count == 1:
        anchor = tf.argmax(tf.abs(loadings[:, 0]), output_type=tf.int32)
        sign = tf.where(
            tf.gather(loadings[:, 0], anchor) >= 0.0,
            tf.constant(1.0, tf.float64),
            tf.constant(-1.0, tf.float64),
        )
        loadings *= sign
        return deviations, loadings, (anchor,)

    anchor_a = tf.argmax(tf.linalg.norm(loadings, axis=1), output_type=tf.int32)
    first = tf.gather(loadings, anchor_a)
    first_norm = tf.maximum(tf.linalg.norm(first), 1.0e-15)
    cosine, sine = first[0] / first_norm, first[1] / first_norm
    rotation = tf.stack(
        (tf.stack((cosine, -sine)), tf.stack((sine, cosine))), axis=0
    )
    loadings = tf.matmul(loadings, rotation)
    second_abs = tf.abs(loadings[:, 1])
    second_abs = tf.tensor_scatter_nd_update(
        second_abs, tf.reshape(anchor_a, [1, 1]), [tf.constant(-1.0, tf.float64)]
    )
    anchor_b = tf.argmax(second_abs, output_type=tf.int32)
    second_sign = tf.where(
        tf.gather(loadings[:, 1], anchor_b) >= 0.0,
        tf.constant(1.0, tf.float64),
        tf.constant(-1.0, tf.float64),
    )
    loadings = loadings * tf.stack((tf.constant(1.0, tf.float64), second_sign))[None, :]
    return deviations, loadings, (anchor_a, anchor_b)


def _encode_state(
    deviations: tf.Tensor,
    loadings: tf.Tensor,
    anchors: tuple[int, ...],
    config: FactorCorrelationGeometryConfig,
) -> tf.Tensor:
    raw_deviations = _softplus_inverse(
        tf.maximum(
            deviations - config.standard_deviation_floor,
            tf.constant(1.0e-12, tf.float64),
        )
    )
    bound = tf.sqrt(tf.constant(1.0 - config.loading_margin, tf.float64))
    if config.factor_count == 1:
        anchor = anchors[0]
        ratio = tf.clip_by_value(loadings[:, 0] / bound, -0.999999, 0.999999)
        raw_loadings = tf.tensor_scatter_nd_update(tf.atanh(ratio), tf.reshape(anchor, [1, 1]),
            tf.reshape(_logit(tf.clip_by_value(tf.gather(ratio, anchor), 1.0e-6, 1.0-1.0e-6)), [1]))
    else:
        anchor_a, anchor_b = anchors
        radius_ratio = tf.clip_by_value(tf.linalg.norm(loadings, axis=1)/bound, 1.0e-6, 1.0-1.0e-6)
        ratio = loadings/bound
        rows = ratio/tf.sqrt(tf.maximum(1.0-tf.reduce_sum(tf.square(ratio), axis=1, keepdims=True), 1.0e-12))
        second = tf.gather(loadings, anchor_b)
        angle_ratio = tf.clip_by_value(tf.atan2(second[1], second[0])/math.pi, 1.0e-6, 1.0-1.0e-6)
        rows = tf.tensor_scatter_nd_update(rows, tf.reshape(tf.stack((anchor_a, anchor_b)), [2, 1]),
            tf.stack((tf.stack((_logit(tf.gather(radius_ratio, anchor_a)), tf.constant(0., tf.float64))),
                      tf.stack((_logit(tf.gather(radius_ratio, anchor_b)), _logit(angle_ratio))))))
        flat = tf.reshape(rows, [-1])
        omitted = 2*anchor_a+1
        indices = tf.range(2*int(loadings.shape[0])-1)
        raw_loadings = tf.gather(flat, indices+tf.cast(indices >= omitted, tf.int32))
    return tf.concat((raw_deviations, raw_loadings), axis=0)


def _weighted_dense_precision(
    offsets: tf.Tensor,
    responses: tf.Tensor,
    weights: tf.Tensor,
    *,
    max_condition_number: float,
    jit_compile: bool = True,
) -> tf.Tensor:
    root_weight = tf.sqrt(weights)[:, None]
    raw = complete_orthogonal_lstsq(offsets * root_weight, responses * root_weight)
    symmetric = 0.5 * (raw + tf.transpose(raw))
    values, vectors = _eigenpairs(symmetric, jit_compile)
    maximum = tf.maximum(tf.reduce_max(tf.abs(values)), 1.0)
    floor = maximum / max_condition_number
    projected = tf.maximum(values, floor)
    return tf.matmul(vectors * projected[None, :], vectors, transpose_b=True)


def _prediction_jacobian_diagnostics(
    raw: tf.Tensor,
    offsets: tf.Tensor,
    *,
    dimension: int,
    anchors: tuple[int, ...],
    config: FactorCorrelationGeometryConfig,
    jit_compile: bool = True,
    active_training_rows: tf.Tensor | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    parameter_count = int(raw.shape[0])
    columns = tf.TensorArray(tf.float64, parameter_count, element_shape=[int(offsets.shape[0])*dimension])
    def column(index, columns):
        with tf.autodiff.ForwardAccumulator(raw, tf.one_hot(index, parameter_count, dtype=raw.dtype)) as tangent:
            covariance, _, _ = _decode_covariance(raw, dimension=dimension, anchors=anchors, config=config)
            precision = tf.linalg.cholesky_solve(tf.linalg.cholesky(covariance), tf.eye(dimension, dtype=tf.float64))
            prediction = tf.reshape(tf.einsum("ij,bj->bi", precision, offsets), [-1])
        return index+1, columns.write(index, tangent.jvp(prediction))
    _, columns = tf.while_loop(lambda index, _: index < parameter_count, column, (tf.constant(0), columns),
                               maximum_iterations=parameter_count, parallel_iterations=1)
    jacobian = tf.transpose(columns.stack())
    if jit_compile:
        upper = tf.linalg.qr(jacobian, full_matrices=False)[1]
        singular = xla_svd(upper, max_iter=100, epsilon=sys.float_info.epsilon, precision_config="").s
    else:
        singular = tf.linalg.svd(jacobian, compute_uv=False)
    largest = tf.reduce_max(singular)
    equation_count = (tf.shape(jacobian)[0] if active_training_rows is None
                      else active_training_rows * dimension)
    tolerance = (
        largest
        * tf.cast(tf.maximum(equation_count, tf.shape(jacobian)[1]), tf.float64)
        * sys.float_info.epsilon
    )
    positive = singular > tolerance
    rank = tf.reduce_sum(tf.cast(positive, tf.int32))
    condition = tf.where(rank > 0, largest/tf.reduce_min(tf.where(positive, singular, float("inf"))), 0.)
    return rank, condition


def _require_matrix(value: tf.Tensor, name: str, dimension: int) -> None:
    if value.shape.rank != 2 or value.shape[0] is None or value.shape[1] != dimension:
        raise ValueError(f"{name} must have static shape [rows, dimension]")


def _softplus_inverse(value: tf.Tensor) -> tf.Tensor:
    return value + tf.math.log(-tf.math.expm1(-value))


def _logit(value: tf.Tensor) -> tf.Tensor:
    return tf.math.log(value) - tf.math.log1p(-value)


def _rejected(
    config: FactorCorrelationGeometryConfig,
    dimension: int,
    status: str,
    *,
    parameter_count: int | None = None,
    anchors: tuple[int, ...] = (),
    diagnostics: Mapping[str, Any] | None = None,
) -> FactorCorrelationGeometryResult:
    count = (
        dimension * 2
        if parameter_count is None and config.factor_count == 1
        else dimension * 3 - 1
        if parameter_count is None
        else int(parameter_count)
    )
    return FactorCorrelationGeometryResult(
        accepted=False,
        status=status,
        factor_count=config.factor_count,
        parameter_count=count,
        anchor_indices=anchors,
        covariance_z=None,
        precision_z=None,
        marginal_standard_deviations=None,
        loadings=None,
        diagnostics={} if diagnostics is None else diagnostics,
    )


def _json_ready(value: Any) -> Any:
    if tf.is_tensor(value) or getattr(value, "__array_interface__", None) is not None:
        return numeric_tensor(value).numpy().tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value
