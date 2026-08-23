"""P4 close-out gate: full canonical score with CHAINED covariance recursion.

The complete analytical recursion over T steps with everything live:
UKF predict tangent (S1) -> flow tangent (S3, per-particle P^i AND dP^i)
-> weight tangent (S4) -> UKF update tangent (S5) -> next step consumes
the posterior covariance AND its tangent -> increment accumulation (S8).
Nonlinear dynamics fixture so every chain link is exercised.
Reset slice: reset_policy='none' equivalent (children carry). S6/S7 wiring
is gated separately in the assembled module test.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_autodiff_oracle_tf import (
    oracle_forward_autodiff_score,
)
from bayesfilter.highdim.ledh_canonical_score_tf import (
    NonlinearScoreModel,
    canonical_value_and_analytical_score,
)

DTYPE = tf.float64


def _model(theta_value: float = 0.6):
    """Nonlinear 2D model: x' = x + theta*sin(x) + noise; z = x + v."""

    def transition_mean_fn(theta, points):
        return points + theta[0] * tf.sin(points)

    def transition_mean_tangent_fn(theta, points, d_points):
        # total tangent: partial_theta (sin x) + (I + theta cos x) d_points
        return tf.sin(points) + d_points + theta[0] * tf.cos(points) * d_points

    def observation_fn(points):
        return points

    def observation_jacobian_fn(points):
        return tf.broadcast_to(
            tf.eye(2, dtype=DTYPE), [tf.shape(points)[0], 2, 2]
        )

    def observation_tangent_fn(points, d_points):
        return d_points

    return NonlinearScoreModel(
        transition_mean_fn=transition_mean_fn,
        transition_mean_tangent_fn=transition_mean_tangent_fn,
        observation_fn=observation_fn,
        observation_jacobian_fn=observation_jacobian_fn,
        observation_tangent_fn=observation_tangent_fn,
        process_covariance=tf.constant(0.4 * np.eye(2), DTYPE),
        observation_covariance=tf.constant(0.6 * np.eye(2), DTYPE),
    )


def test_full_chained_score_matches_oracle_nonlinear():
    rng = np.random.default_rng(61)
    n, dim, horizon = 6, 2, 3
    initial = tf.constant(rng.standard_normal((n, dim)), DTYPE)
    covs = tf.constant(np.stack([np.eye(dim)] * n), DTYPE)
    noises = tf.constant(rng.standard_normal((horizon, n, dim)), DTYPE)
    observations = tf.constant(rng.standard_normal((horizon, dim)), DTYPE)
    theta0 = tf.constant([0.6], DTYPE)
    model = _model()

    def value_fn(theta):
        value, _ = canonical_value_and_analytical_score(
            model, theta, initial, covs, noises, observations,
            substeps=10, with_score=False,
        )
        return value

    oracle = oracle_forward_autodiff_score(value_fn, theta0)
    value, score = canonical_value_and_analytical_score(
        model, theta0, initial, covs, noises, observations,
        substeps=10, with_score=True,
    )
    assert np.isfinite(float(value.numpy()))
    err = abs(float(score[0].numpy()) - float(oracle[0].numpy()))
    scale = max(abs(float(oracle[0].numpy())), 1.0)
    assert err < 1.0e-4 * scale, (
        f"full chained analytical {float(score[0].numpy())} vs oracle "
        f"{float(oracle[0].numpy())}"
    )


def test_full_program_score_with_reset_and_dualcap_matches_oracle():
    """Q1.1 gate (S6+S7): the COMPLETE per-step program — flow, weight,
    Sinkhorn+Contract-E reset, dual-cap trust-region correction — with the
    analytical score, vs the autodiff oracle. Nonlinear fixture, small
    scope for CPU speed; rtol 1e-4 declared."""

    rng = np.random.default_rng(71)
    n, dim, horizon = 16, 2, 2
    initial = tf.constant(rng.standard_normal((n, dim)), DTYPE)
    covs = tf.constant(np.stack([np.eye(dim)] * n), DTYPE)
    noises = tf.constant(rng.standard_normal((horizon, n, dim)), DTYPE)
    observations = tf.constant(rng.standard_normal((horizon, dim)), DTYPE)
    theta0 = tf.constant([0.6], DTYPE)
    model = _model()
    # deterministic +/- unit design rows, N divisible by 2*dim
    base = np.concatenate([np.eye(dim), -np.eye(dim)], axis=0)
    design = tf.constant(
        np.tile(base, (n // (2 * dim), 1)), DTYPE
    )

    kwargs = dict(
        substeps=8,
        reset_policy="contract_e",
        reset_design=design,
        reset_sinkhorn_steps=4,
        reset_balance_steps=2,
        correction_steps=2,
        pairwise_steps=1,
        coordinate_cap=0.98,
    )

    def value_fn(theta):
        value, _ = canonical_value_and_analytical_score(
            model, theta, initial, covs, noises, observations,
            with_score=False, **kwargs,
        )
        return value

    oracle = oracle_forward_autodiff_score(value_fn, theta0)
    value, score = canonical_value_and_analytical_score(
        model, theta0, initial, covs, noises, observations,
        with_score=True, **kwargs,
    )
    assert np.isfinite(float(value.numpy()))
    err = abs(float(score[0].numpy()) - float(oracle[0].numpy()))
    scale = max(abs(float(oracle[0].numpy())), 1.0)
    assert err < 1.0e-4 * scale, (
        f"full-program analytical {float(score[0].numpy())} vs oracle "
        f"{float(oracle[0].numpy())}"
    )
