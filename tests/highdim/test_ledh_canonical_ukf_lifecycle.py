"""Conformance tests C-1, C-6, C-7 + Kalman fixture for the UKF lifecycle.

Written BEFORE the implementation per execution-plan rule R-C. These tests
import `ledh_ukf_lifecycle_tf`, which does not exist yet at authoring time —
the suite must fail first, then pass when P1 lands, proving the gate is
real. Tolerances are the execution plan's declared values (R-E).
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

from bayesfilter.highdim.ledh_ukf_lifecycle_tf import (
    triple_gather,
    ukf_predict_per_particle,
    ukf_update_per_particle,
)

DTYPE = tf.float64
ATOL_LINEAR = 1.0e-10  # float64 linear-Gaussian: unscented == exact


def _linear_fixture(seed: int, n_particles: int = 5, dim: int = 3):
    rng = np.random.default_rng(seed)
    transition = np.eye(dim) + 0.1 * rng.standard_normal((dim, dim))
    process_cov = np.eye(dim) * 0.5 + 0.1 * np.ones((dim, dim))
    process_cov = process_cov @ process_cov.T / 2.0
    states = rng.standard_normal((n_particles, dim))
    covs = np.stack(
        [np.eye(dim) * (0.5 + 0.5 * i) for i in range(n_particles)]
    )
    return (
        tf.constant(transition, DTYPE),
        tf.constant(process_cov, DTYPE),
        tf.constant(states, DTYPE),
        tf.constant(covs, DTYPE),
    )


def test_c1_predict_is_per_particle_and_matches_kalman_on_linear():
    transition, process_cov, states, covs = _linear_fixture(3)

    def mean_fn(points):
        return tf.linalg.matvec(
            tf.broadcast_to(transition, [tf.shape(points)[0], 3, 3]), points
        )

    means, predicted = ukf_predict_per_particle(
        states, covs, mean_fn, process_cov
    )
    # (a) exists per particle, (b) differs across particles
    assert predicted.shape == covs.shape
    spread = tf.reduce_max(
        tf.math.reduce_std(predicted, axis=0)
    ).numpy()
    assert spread > 1.0e-6, "predicted covariances identical across particles"
    # (c) equals exact Kalman prediction F P F^T + Q on the linear fixture
    expected = (
        tf.einsum("ij,njk,lk->nil", transition, covs, transition)
        + process_cov[None]
    )
    err = tf.reduce_max(tf.abs(predicted - expected)).numpy()
    assert err < ATOL_LINEAR, f"UKF prediction differs from Kalman: {err}"
    mean_err = tf.reduce_max(
        tf.abs(means - tf.linalg.matvec(transition[None], states))
    ).numpy()
    assert mean_err < ATOL_LINEAR


def test_c1_rejects_identity_shortcut():
    """The wiring test that the 2026-08 placeholder could never pass:
    a genuinely nonlinear mean_fn must produce state-dependent covariances."""
    _, process_cov, states, covs = _linear_fixture(5)

    def nonlinear_mean_fn(points):
        return points + 0.3 * tf.square(points)

    _, predicted = ukf_predict_per_particle(
        states, covs, nonlinear_mean_fn, process_cov
    )
    # Different ancestors (different states AND covs) must give different
    # predicted covariances; identity/shared shortcut would make them equal.
    pairwise = tf.reduce_max(
        tf.abs(predicted[0] - predicted[1])
    ).numpy()
    assert pairwise > 1.0e-4, "prediction insensitive to ancestor state"


def test_c6_update_recursion_two_step_chain():
    transition, process_cov, states, covs = _linear_fixture(7)
    obs_matrix = tf.constant(
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), DTYPE
    )
    obs_cov = tf.eye(2, dtype=DTYPE) * 0.3

    def mean_fn(points):
        return tf.linalg.matvec(
            tf.broadcast_to(transition, [tf.shape(points)[0], 3, 3]), points
        )

    def obs_fn(points):
        return tf.linalg.matvec(
            tf.broadcast_to(obs_matrix, [tf.shape(points)[0], 2, 3]), points
        )

    means, predicted = ukf_predict_per_particle(
        states, covs, mean_fn, process_cov
    )
    observation = tf.constant(np.array([0.4, -0.2]), DTYPE)
    post_means, posterior = ukf_update_per_particle(
        means, predicted, obs_fn, obs_cov, observation
    )
    # Update must shrink covariance in the observed subspace
    observed_var_before = tf.einsum(
        "oi,nij,pj->nop", obs_matrix, predicted, obs_matrix
    )
    observed_var_after = tf.einsum(
        "oi,nij,pj->nop", obs_matrix, posterior, obs_matrix
    )
    assert bool(
        tf.reduce_all(
            tf.linalg.diag_part(observed_var_after)
            < tf.linalg.diag_part(observed_var_before)
        ).numpy()
    ), "update did not contract observed subspace"
    # Exact Kalman check per particle (linear fixture)
    for i in range(5):
        p = predicted[i].numpy()
        h = obs_matrix.numpy()
        s = h @ p @ h.T + obs_cov.numpy()
        k = p @ h.T @ np.linalg.inv(s)
        expected = (np.eye(3) - k @ h) @ p
        err = np.max(np.abs(posterior[i].numpy() - expected))
        assert err < 1.0e-8, f"particle {i}: update differs from Kalman ({err})"
    # Recursion: second step must consume the posterior, not the prior
    _, second = ukf_predict_per_particle(
        post_means, posterior, mean_fn, process_cov
    )
    _, second_wrong = ukf_predict_per_particle(
        post_means, predicted, mean_fn, process_cov
    )
    assert (
        tf.reduce_max(tf.abs(second - second_wrong)).numpy() > 1.0e-6
    ), "chaining test cannot distinguish posterior from prior input"


def test_c7_triple_moves_together():
    _, _, states, covs = _linear_fixture(11)
    weights = tf.constant(
        np.array([0.1, 0.2, 0.3, 0.25, 0.15]), DTYPE
    )
    # Mark particle 3's covariance with a recognizable scale
    marked = tf.tensor_scatter_nd_update(
        covs, [[3]], [tf.eye(3, dtype=DTYPE) * 777.0]
    )
    indices = tf.constant([3, 3, 0, 1, 2], tf.int32)
    new_states, new_covs, new_weights = triple_gather(
        states, marked, weights, indices
    )
    # The marked covariance must follow its state
    assert abs(float(new_covs[0, 0, 0]) - 777.0) < 1e-12
    assert abs(float(new_covs[1, 0, 0]) - 777.0) < 1e-12
    assert bool(
        tf.reduce_all(tf.equal(new_states[0], states[3])).numpy()
    )
    assert bool(
        tf.reduce_all(tf.equal(new_weights, tf.gather(weights, indices))).numpy()
    )
