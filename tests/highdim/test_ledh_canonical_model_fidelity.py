"""Model-fidelity gates (added 2026-08-23 after the KSC defect class).

These verify the CANONICAL onboardings against INDEPENDENT reference
implementations of the model definitions — the gate class the oracle
self-consistency gates cannot cover (they verify whatever model was
defined; these verify the model IS the reference).
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

DTYPE = tf.float64


def test_generalized_sv_densities_match_native_reference():
    """Observation and transition log-densities of the onboarded model
    must equal `NativeGeneralizedSVSSM`'s on shared points (float64,
    atol 1e-10)."""

    from bayesfilter.highdim.ledh_canonical_models_tf import (
        generalized_sv_canonical_model,
    )
    from bayesfilter.highdim.native_generalized_sv import (
        NativeGeneralizedSVSSM,
    )

    native = NativeGeneralizedSVSSM()
    theta_np = [
        float(np.arctanh(0.9)), float(np.arctanh(0.8)), -0.5, -0.7, 0.2,
    ]
    theta = tf.constant(theta_np, DTYPE)
    model, _sd = generalized_sv_canonical_model(theta)
    rng = np.random.default_rng(311)
    points = tf.constant(rng.standard_normal((16, 2)), DTYPE)
    observation = tf.constant([0.7], DTYPE)

    mine = model.observation_log_density_fn(theta, points, observation)
    reference = native.observation_log_density(
        theta, points, observation, 0
    )
    err = float(tf.reduce_max(tf.abs(mine - reference)).numpy())
    assert err < 1.0e-10, f"observation density infidelity: {err}"

    # transition density: canonical weight uses N(x'; mean_fn(x), Q)
    ancestors = tf.constant(rng.standard_normal((16, 2)), DTYPE)
    children = tf.constant(rng.standard_normal((16, 2)), DTYPE)
    mean = model.transition_mean_fn(theta, ancestors)
    q = model.process_covariance
    chol = tf.linalg.cholesky(q)
    residual = children - mean
    solved = tf.linalg.triangular_solve(
        tf.broadcast_to(chol, [16, 2, 2]), residual[:, :, None]
    )[:, :, 0]
    mine_trans = -0.5 * (
        tf.reduce_sum(tf.square(solved), axis=1)
        + tf.constant(2.0 * np.log(2.0 * np.pi), DTYPE)
        + 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
    )
    reference_trans = native.transition_log_density(
        theta, ancestors, children, 0
    )
    err_t = float(tf.reduce_max(tf.abs(mine_trans - reference_trans)).numpy())
    assert err_t < 1.0e-10, f"transition density infidelity: {err_t}"


def test_diagonal_lgssm_observation_matrix_matches_reference():
    """The frozen target's `_LGSSM_MATRIX` constants (previously replaced
    by eye(3) — fidelity defect)."""

    from bayesfilter.highdim.ledh_canonical_models_tf import (
        diagonal_lgssm_canonical_model,
    )

    reference_matrix = np.array(
        [[1.0, 0.25, -0.15], [0.2, 1.1, 0.3], [-0.1, 0.35, 0.9]]
    )
    theta = tf.constant([0.9, 0.8, 0.7, 0.6, 0.8], DTYPE)
    model, _sd = diagonal_lgssm_canonical_model(theta)
    probe = tf.constant(np.eye(3), DTYPE)  # unit states
    observed = model.observation_fn(probe).numpy()
    # observation_fn(e_i) returns column i of the matrix
    assert np.allclose(observed.T, reference_matrix, atol=1e-12), observed


def test_austria_observation_variance_matches_reference_form():
    """Austria: observation covariance must be 100*exp(2*theta_2)*I_9 and
    the extraction must select the infectious (odd) compartments —
    verified against the model definition constants."""

    from bayesfilter.highdim.ledh_canonical_models_tf import (
        austria_sir_canonical_model,
    )

    theta = tf.constant([0.0, 0.0, 0.3], DTYPE)
    model, _sd = austria_sir_canonical_model(theta)
    expected_variance = 100.0 * np.exp(2.0 * 0.3)
    diag = tf.linalg.diag_part(model.observation_covariance).numpy()
    assert np.allclose(diag, expected_variance, rtol=1e-12)
    probe = tf.constant(np.eye(18), DTYPE)
    observed = model.observation_fn(probe).numpy()
    for o in range(9):
        picked = np.argmax(observed[:, o])
        assert picked == 2 * o + 1, (
            f"observation row {o} extracts state {picked}, expected "
            f"{2 * o + 1} (infectious compartments)"
        )


def test_predator_prey_noise_scales_match_reference():
    """Reference adapter: transition adds 2*noise (Q=4I); observation
    residual/4 with 2*log(4) normalizer (R=4I)."""

    from bayesfilter.highdim.ledh_canonical_models_tf import (
        predator_prey_canonical_model,
    )

    theta = tf.constant([0.8, 90.0, 25.0, 0.5, 0.4, 0.3], DTYPE)
    model, _sd = predator_prey_canonical_model(theta)
    assert np.allclose(
        model.process_covariance.numpy(), 4.0 * np.eye(2), atol=1e-12
    )
    assert np.allclose(
        model.observation_covariance.numpy(), 4.0 * np.eye(2), atol=1e-12
    )
