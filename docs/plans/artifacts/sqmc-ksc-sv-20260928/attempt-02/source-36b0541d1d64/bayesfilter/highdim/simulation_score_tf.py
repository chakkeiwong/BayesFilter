"""Simulation-based observed-data score estimators.

The primary estimator is the Fisher identity evaluated with latent paths drawn
from the model prior.  Callers supply model-specific simulation and complete
data score code; this module owns only the weighting, validation, and
diagnostic contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import tensorflow as tf


@dataclass(frozen=True)
class SimulationScoreEstimate:
    """Self-normalized Fisher-identity estimate and reliability diagnostics."""

    log_marginal: tf.Tensor
    score: tf.Tensor
    normalized_weights: tf.Tensor
    effective_sample_size: tf.Tensor
    effective_sample_fraction: tf.Tensor
    maximum_normalized_weight: tf.Tensor
    log_weight_range: tf.Tensor
    finite: tf.Tensor
    collapsed: tf.Tensor

    def as_dict(self) -> dict[str, Any]:
        """Return tensors in a stable artifact-friendly mapping."""

        return {
            "log_marginal": self.log_marginal,
            "score": self.score,
            "normalized_weights": self.normalized_weights,
            "effective_sample_size": self.effective_sample_size,
            "effective_sample_fraction": self.effective_sample_fraction,
            "maximum_normalized_weight": self.maximum_normalized_weight,
            "log_weight_range": self.log_weight_range,
            "finite": self.finite,
            "collapsed": self.collapsed,
        }


def fisher_identity_simulation_score(
    log_observation_likelihood: tf.Tensor,
    complete_data_score: tf.Tensor,
    *,
    minimum_effective_sample_fraction: float = 0.01,
) -> SimulationScoreEstimate:
    """Estimate ``grad log p(y)`` from prior-simulated latent paths.

    ``log_observation_likelihood[m]`` is ``log p(y | x_m)`` and
    ``complete_data_score[m]`` is the score of ``log p(theta, x_m, y)`` with
    respect to the same parameterization.  Since the paths are sampled from
    the latent prior at ``theta``, self-normalization gives the Monte Carlo
    approximation to ``E[complete_score | y]``.
    """

    if isinstance(minimum_effective_sample_fraction, bool):
        raise ValueError("minimum_effective_sample_fraction must be numeric")
    threshold = float(minimum_effective_sample_fraction)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("minimum_effective_sample_fraction must be in [0, 1]")

    log_like = tf.convert_to_tensor(log_observation_likelihood)
    score = tf.convert_to_tensor(complete_data_score)
    if log_like.shape.rank != 1:
        raise ValueError("log_observation_likelihood must have shape [paths]")
    if score.shape.rank != 2:
        raise ValueError("complete_data_score must have shape [paths, parameters]")
    if log_like.shape[0] is not None and score.shape[0] is not None:
        if log_like.shape[0] != score.shape[0]:
            raise ValueError("likelihood and score path counts differ")
    if score.shape[0] == 0:
        raise ValueError("at least one simulated path is required")

    log_like = tf.cast(log_like, tf.float64)
    score = tf.cast(score, tf.float64)
    path_count = tf.cast(tf.shape(log_like)[0], tf.float64)
    finite = tf.reduce_all(tf.math.is_finite(log_like)) & tf.reduce_all(
        tf.math.is_finite(score)
    )
    log_normalizer = tf.reduce_logsumexp(log_like)
    normalized_weights = tf.nn.softmax(log_like)
    estimate = tf.reduce_sum(normalized_weights[:, None] * score, axis=0)
    effective_sample_size = tf.math.reciprocal(
        tf.reduce_sum(tf.square(normalized_weights))
    )
    effective_sample_fraction = effective_sample_size / path_count
    maximum_normalized_weight = tf.reduce_max(normalized_weights)
    log_weight_range = tf.reduce_max(log_like) - tf.reduce_min(log_like)
    collapsed = effective_sample_fraction < tf.cast(threshold, tf.float64)
    return SimulationScoreEstimate(
        log_marginal=log_normalizer - tf.math.log(path_count),
        score=estimate,
        normalized_weights=normalized_weights,
        effective_sample_size=effective_sample_size,
        effective_sample_fraction=effective_sample_fraction,
        maximum_normalized_weight=maximum_normalized_weight,
        log_weight_range=log_weight_range,
        finite=finite,
        collapsed=collapsed,
    )


__all__ = [
    "SimulationScoreEstimate",
    "fisher_identity_simulation_score",
]
