"""P4 S1/S5 gate: analytical UKF predict/update parameter tangents.

The covariance-recursion tangent: d(P^i_pred)/dtheta through sigma points
(Cholesky differential via the Phi-operator identity) and d(P^i_post)/dtheta
through the update. Gated against the autodiff oracle on a NONLINEAR
dynamics fixture (so the sigma-point chain rule is genuinely exercised —
linear fixtures would let errors in the Cholesky differential cancel).
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_autodiff_oracle_tf import (
    oracle_forward_autodiff_score,
)
from bayesfilter.highdim.ledh_canonical_score_stages_tf import (
    ukf_predict_with_parameter_tangent,
    ukf_update_with_parameter_tangent,
)

DTYPE = tf.float64


def _fixture(seed: int, n: int = 4, dim: int = 2):
    rng = np.random.default_rng(seed)
    states = tf.constant(rng.standard_normal((n, dim)), DTYPE)
    covs = tf.constant(
        np.stack([np.eye(dim) * (0.5 + 0.2 * i) for i in range(n)]), DTYPE
    )
    d_states = tf.constant(rng.standard_normal((n, dim)), DTYPE)
    d_covs = tf.zeros_like(covs)
    q = tf.constant(0.4 * np.eye(dim), DTYPE)
    return states, covs, d_states, d_covs, q


def test_s1_predict_tangent_matches_oracle_nonlinear():
    states, covs, d_states, d_covs, q = _fixture(51)
    theta0 = tf.constant([0.7], DTYPE)

    def mean_fn_theta(theta):
        def mean_fn(points):
            return points + theta[0] * tf.square(points)

        return mean_fn

    def d_mean_fn(points, d_points):
        # total tangent: partial_theta + jacobian @ d_points
        # f = x + theta x^2; df/dtheta = x^2; df/dx = I + 2 theta x
        return tf.square(points) + (
            d_points + 2.0 * theta0[0] * points * d_points
        )

    def summary_fn(theta_vec):
        theta = theta_vec[0]

        def mean_fn(points):
            return points + theta * tf.square(points)

        # theta also perturbs the input states (chained tangent scenario)
        # NOTE: 0.7 as a PYTHON constant, not theta0[0] — the watched
        # tensor slice would carry its own accumulator tangent and
        # cancel the shift path's derivative (correctly, but that is
        # not the function under test). Recorded oracle-usage lesson.
        shifted = states + (theta - 0.7) * d_states
        from bayesfilter.highdim.ledh_ukf_lifecycle_tf import (
            ukf_predict_per_particle,
        )

        means, predicted = ukf_predict_per_particle(
            shifted, covs, mean_fn, q
        )
        return tf.reduce_sum(tf.sin(means)) + tf.reduce_sum(
            tf.cos(predicted)
        )

    oracle = oracle_forward_autodiff_score(summary_fn, theta0)

    mean_fn = mean_fn_theta(theta0)
    means, predicted, d_means, d_predicted, valid = (
        ukf_predict_with_parameter_tangent(
            states, covs, d_states, d_covs, mean_fn, d_mean_fn, q
        )
    )
    assert valid.numpy().all(), "Sigma points generation should be valid"
    analytical = tf.reduce_sum(tf.cos(means) * d_means) - tf.reduce_sum(
        tf.sin(predicted) * d_predicted
    )
    err = abs(float(analytical.numpy()) - float(oracle[0].numpy()))
    scale = max(abs(float(oracle[0].numpy())), 1.0)
    assert err < 1.0e-4 * scale, (
        f"S1 analytical {float(analytical.numpy())} vs oracle "
        f"{float(oracle[0].numpy())}"
    )


def test_s5_update_tangent_matches_oracle():
    states, covs, d_states, d_covs, q = _fixture(53)
    theta0 = tf.constant([0.7], DTYPE)
    obs_matrix = tf.constant(np.eye(2), DTYPE)
    obs_cov = tf.constant(0.6 * np.eye(2), DTYPE)
    observation = tf.constant(np.array([0.3, -0.4]), DTYPE)

    def obs_fn(points):
        return tf.linalg.matvec(
            tf.broadcast_to(obs_matrix, [tf.shape(points)[0], 2, 2]), points
        )

    def d_obs_fn(points, d_points):
        return tf.linalg.matvec(
            tf.broadcast_to(obs_matrix, [tf.shape(points)[0], 2, 2]),
            d_points,
        )

    def summary_fn(theta):
        shifted = states + (theta[0] - 0.7) * d_states
        from bayesfilter.highdim.ledh_ukf_lifecycle_tf import (
            ukf_update_per_particle,
        )

        post_means, posterior = ukf_update_per_particle(
            shifted, covs, obs_fn, obs_cov, observation
        )
        return tf.reduce_sum(tf.sin(post_means)) + tf.reduce_sum(
            tf.cos(posterior)
        )

    oracle = oracle_forward_autodiff_score(summary_fn, theta0)
    post_means, posterior, d_post_means, d_posterior, valid = (
        ukf_update_with_parameter_tangent(
            states,
            covs,
            d_states,
            d_covs,
            obs_fn,
            d_obs_fn,
            obs_cov,
            observation,
        )
    )
    assert valid.numpy().all(), "Sigma points generation should be valid"
    analytical = tf.reduce_sum(
        tf.cos(post_means) * d_post_means
    ) - tf.reduce_sum(tf.sin(posterior) * d_posterior)
    err = abs(float(analytical.numpy()) - float(oracle[0].numpy()))
    scale = max(abs(float(oracle[0].numpy())), 1.0)
    assert err < 1.0e-4 * scale, (
        f"S5 analytical {float(analytical.numpy())} vs oracle "
        f"{float(oracle[0].numpy())}"
    )
