"""Experimental opt-in batched value+score interface candidates.

This module is intentionally not re-exported from :mod:`bayesfilter`.  It is a
public import path only for explicit experimental use while the batch-over-
parameters contract is being evaluated.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import tensorflow as tf

from bayesfilter.linear.experimental_batched_kalman_tf import (
    tf_batched_kalman_value_and_score,
)
from bayesfilter.nonlinear.experimental_batched_svd_sigma_point_tf import (
    TFBatchedSVDBackend,
    TFBatchedStructuralFirstDerivatives,
    TFBatchedStructuralStateSpace,
    tf_batched_svd_sigma_point_value_and_score,
)


ExperimentalBatchedBackend = Literal[
    "tf_batched_kalman",
    "tf_svd_ukf",
    "tf_svd_cubature",
    "tf_principal_sqrt_ukf",
]

NONCLAIMS = (
    "experimental opt-in interface only",
    "no production default readiness claim",
    "no GPU performance claim",
    "no HMC or NeuTra integration claim",
)


@dataclass(frozen=True)
class ExperimentalBatchedValueScoreMetadata:
    """Metadata for an experimental batched value+score call."""

    backend: str
    batch_size: int
    parameter_dim: int
    experimental: bool
    scalar_fallback_used: bool
    nonclaims: tuple[str, ...] = NONCLAIMS


@dataclass(frozen=True)
class ExperimentalBatchedValueScoreResult:
    """Batched value+score tensors plus explicit experimental metadata."""

    value: tf.Tensor
    score: tf.Tensor
    metadata: ExperimentalBatchedValueScoreMetadata
    diagnostics: Mapping[str, Any]


def _metadata(
    *,
    backend: str,
    value: tf.Tensor,
    score: tf.Tensor,
    scalar_fallback_used: bool,
) -> ExperimentalBatchedValueScoreMetadata:
    value = tf.convert_to_tensor(value, dtype=tf.float64)
    score = tf.convert_to_tensor(score, dtype=tf.float64)
    if value.shape.rank != 1:
        raise ValueError("batched value must have rank 1")
    if score.shape.rank != 2:
        raise ValueError("batched score must have rank 2")
    batch_size = value.shape[0]
    parameter_dim = score.shape[1]
    if batch_size is None or parameter_dim is None:
        raise ValueError("batched value and score require static batch/parameter dimensions")
    if score.shape[0] != batch_size:
        raise ValueError("score batch dimension must match value batch dimension")
    return ExperimentalBatchedValueScoreMetadata(
        backend=backend,
        batch_size=int(batch_size),
        parameter_dim=int(parameter_dim),
        experimental=True,
        scalar_fallback_used=scalar_fallback_used,
    )


def experimental_batched_kalman_value_score(
    observations: tf.Tensor,
    *,
    transition_offset: tf.Tensor,
    transition_matrix: tf.Tensor,
    transition_covariance: tf.Tensor,
    observation_offset: tf.Tensor,
    observation_matrix: tf.Tensor,
    observation_covariance: tf.Tensor,
    initial_state_mean: tf.Tensor,
    initial_state_covariance: tf.Tensor,
    d_initial_state_mean: tf.Tensor,
    d_initial_state_covariance: tf.Tensor,
    d_transition_offset: tf.Tensor,
    d_transition_matrix: tf.Tensor,
    d_transition_covariance: tf.Tensor,
    d_observation_offset: tf.Tensor,
    d_observation_matrix: tf.Tensor,
    d_observation_covariance: tf.Tensor,
    jitter: tf.Tensor | float = 0.0,
) -> ExperimentalBatchedValueScoreResult:
    """Return batched Kalman value+score with experimental metadata."""

    value, score = tf_batched_kalman_value_and_score(
        observations,
        transition_offset=transition_offset,
        transition_matrix=transition_matrix,
        transition_covariance=transition_covariance,
        observation_offset=observation_offset,
        observation_matrix=observation_matrix,
        observation_covariance=observation_covariance,
        initial_state_mean=initial_state_mean,
        initial_state_covariance=initial_state_covariance,
        d_initial_state_mean=d_initial_state_mean,
        d_initial_state_covariance=d_initial_state_covariance,
        d_transition_offset=d_transition_offset,
        d_transition_matrix=d_transition_matrix,
        d_transition_covariance=d_transition_covariance,
        d_observation_offset=d_observation_offset,
        d_observation_matrix=d_observation_matrix,
        d_observation_covariance=d_observation_covariance,
        jitter=jitter,
    )
    return ExperimentalBatchedValueScoreResult(
        value=value,
        score=score,
        metadata=_metadata(
            backend="tf_batched_kalman",
            value=value,
            score=score,
            scalar_fallback_used=False,
        ),
        diagnostics={"path": "experimental_batched_kalman"},
    )


def experimental_batched_svd_sigma_point_value_score(
    observations: tf.Tensor,
    model: TFBatchedStructuralStateSpace,
    derivatives: TFBatchedStructuralFirstDerivatives,
    *,
    backend: TFBatchedSVDBackend,
    placement_floor: tf.Tensor | float = 0.0,
    innovation_floor: tf.Tensor | float = 1.0e-12,
    rank_tolerance: tf.Tensor | float = 1.0e-12,
    spectral_gap_tolerance: tf.Tensor | float = 1.0e-8,
    fixed_null_tolerance: tf.Tensor | float = 1.0e-10,
    jitter: tf.Tensor | float = 0.0,
    allow_fixed_null_support: bool = False,
) -> ExperimentalBatchedValueScoreResult:
    """Return batched SVD sigma-point value+score with experimental metadata."""

    if backend not in ("tf_svd_ukf", "tf_svd_cubature", "tf_principal_sqrt_ukf"):
        raise ValueError(f"unsupported experimental batched SVD backend: {backend}")
    value, score, diagnostics = tf_batched_svd_sigma_point_value_and_score(
        observations,
        model,
        derivatives,
        backend=backend,
        placement_floor=placement_floor,
        innovation_floor=innovation_floor,
        rank_tolerance=rank_tolerance,
        spectral_gap_tolerance=spectral_gap_tolerance,
        fixed_null_tolerance=fixed_null_tolerance,
        jitter=jitter,
        allow_fixed_null_support=allow_fixed_null_support,
    )
    return ExperimentalBatchedValueScoreResult(
        value=value,
        score=score,
        metadata=_metadata(
            backend=backend,
            value=value,
            score=score,
            scalar_fallback_used=False,
        ),
        diagnostics=dict(diagnostics),
    )


def experimental_scalar_callback_fallback_value_score(
    theta_batch: tf.Tensor,
    scalar_value_score: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    *,
    backend: str = "explicit_scalar_callback",
) -> ExperimentalBatchedValueScoreResult:
    """Stack row-wise values/scores from an explicit scalar callback."""

    theta_batch = tf.convert_to_tensor(theta_batch, dtype=tf.float64)
    if theta_batch.shape.rank != 2:
        raise ValueError("theta_batch must have rank 2")
    values = []
    scores = []
    for theta in tf.unstack(theta_batch, axis=0):
        value, score = scalar_value_score(theta)
        values.append(tf.reshape(tf.convert_to_tensor(value, dtype=tf.float64), []))
        scores.append(tf.convert_to_tensor(score, dtype=tf.float64))
    value_batch = tf.stack(values, axis=0)
    score_batch = tf.stack(scores, axis=0)
    return ExperimentalBatchedValueScoreResult(
        value=value_batch,
        score=score_batch,
        metadata=_metadata(
            backend=backend,
            value=value_batch,
            score=score_batch,
            scalar_fallback_used=True,
        ),
        diagnostics={
            "path": "explicit_scalar_callback_fallback",
            "row_count": int(theta_batch.shape[0]),
        },
    )
