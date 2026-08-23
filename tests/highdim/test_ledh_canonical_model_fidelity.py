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


def test_austria_dynamics_match_vendored_reference_adapter():
    """Differential gate vs the ORIGINAL author's adapter (vendored from
    git at 43de3cb6^): transition mean (zero-noise RK4 push) and the
    observation quadratic form must agree on shared inputs. float32
    reference => rtol 1e-5."""

    import vendored_reference_batch_adapters as reference

    from bayesfilter.highdim.ledh_canonical_models_tf import (
        austria_sir_canonical_model,
    )

    theta_np = [0.1, -0.2, 0.15]
    theta64 = tf.constant(theta_np, DTYPE)
    model, _sd = austria_sir_canonical_model(theta64)
    adapter = reference.parameterized_austria_sir_batch_adapter()
    rng = np.random.default_rng(501)
    from bayesfilter.highdim.models import zhao_cui_sir_austria_model

    initial_mean = zhao_cui_sir_austria_model().initial_mean.numpy()
    particles_np = initial_mean[None, :] + 0.5 * rng.standard_normal((12, 18))
    theta32 = tf.constant([theta_np], tf.float32)
    particles32 = tf.constant(particles_np[None, :, :], tf.float32)
    zero_noise = tf.zeros([12, 18], tf.float32)
    reference_push = adapter.transition_value(
        theta32, particles32, zero_noise, 0
    )[0].numpy()
    mine_push = model.transition_mean_fn(
        theta64, tf.constant(particles_np, DTYPE)
    ).numpy()
    rel = np.max(
        np.abs(mine_push - reference_push)
        / np.maximum(np.abs(reference_push), 1.0)
    )
    assert rel < 1.0e-5, f"Austria transition-mean infidelity: rel {rel}"

    observation_np = rng.standard_normal(9).astype(np.float32)
    reference_obs = adapter.observation_value(
        theta32, particles32, tf.constant(observation_np), 0
    )[0].numpy()
    observed = model.observation_fn(tf.constant(particles_np, DTYPE)).numpy()
    variance = 100.0 * np.exp(2.0 * theta_np[2])
    mine_obs = -0.5 * (
        np.sum(
            (observation_np[None, :].astype(np.float64) - observed) ** 2,
            axis=1,
        )
        / variance
        + 9.0 * np.log(2.0 * np.pi * variance)
    )
    rel_obs = np.max(
        np.abs(mine_obs - reference_obs)
        / np.maximum(np.abs(reference_obs), 1.0)
    )
    assert rel_obs < 1.0e-4, (
        f"Austria observation-density infidelity: rel {rel_obs}"
    )


def test_predator_prey_dynamics_match_vendored_reference_adapter():
    """Differential gate vs the vendored original adapter: RK4 push and
    observation density on shared inputs (float32 reference)."""

    import vendored_reference_batch_adapters as reference

    from bayesfilter.highdim.ledh_canonical_models_tf import (
        predator_prey_canonical_model,
    )

    theta_np = [0.8, 90.0, 25.0, 0.5, 0.4, 0.3]
    theta64 = tf.constant(theta_np, DTYPE)
    model, _sd = predator_prey_canonical_model(theta64)
    adapter = reference.predator_prey_batch_adapter()
    rng = np.random.default_rng(503)
    particles_np = np.abs(
        np.array([50.0, 5.0])[None, :] + 2.0 * rng.standard_normal((12, 2))
    )
    theta32 = tf.constant([theta_np], tf.float32)
    particles32 = tf.constant(particles_np[None, :, :], tf.float32)
    zero_noise = tf.zeros([12, 2], tf.float32)
    reference_push = adapter.transition_value(
        theta32, particles32, zero_noise, 0
    )[0].numpy()
    mine_push = model.transition_mean_fn(
        theta64, tf.constant(particles_np, DTYPE)
    ).numpy()
    rel = np.max(
        np.abs(mine_push - reference_push)
        / np.maximum(np.abs(reference_push), 1.0)
    )
    assert rel < 1.0e-4, f"predator-prey transition infidelity: rel {rel}"

    observation_np = rng.standard_normal(2).astype(np.float32) + np.array(
        [50.0, 5.0], np.float32
    )
    reference_obs = adapter.observation_value(
        theta32, particles32, tf.constant(observation_np), 0
    )[0].numpy()
    mine_obs = -0.5 * (
        np.sum(
            (observation_np[None, :].astype(np.float64) - particles_np) ** 2,
            axis=1,
        )
        / 4.0
        + 2.0 * np.log(4.0)
        + 2.0 * np.log(2.0 * np.pi)
    )
    rel_obs = np.max(
        np.abs(mine_obs - reference_obs)
        / np.maximum(np.abs(reference_obs), 1.0)
    )
    assert rel_obs < 1.0e-4, (
        f"predator-prey observation-density infidelity: rel {rel_obs}"
    )


def test_ksc_mixture_density_matches_vendored_reference():
    """KSC weight density must equal the vendored original adapter's
    mixture logsumexp on shared inputs (class-2 referent; float32
    reference => rtol 1e-4). Closes the KSC independent-fidelity cell —
    the previous moment-matched-Gaussian weight was a residual
    infidelity flagged PENDING in the registry."""

    import vendored_reference_batch_adapters as reference

    from bayesfilter.highdim.ledh_canonical_models_tf import (
        ksc_sv_canonical_model,
    )

    theta_np = [0.5, 0.1]
    theta64 = tf.constant(theta_np, DTYPE)
    model, _sd = ksc_sv_canonical_model(theta64)
    adapter = reference.ksc_mixture_sv_batch_adapter()
    rng = np.random.default_rng(521)
    h_np = rng.standard_normal(16)
    observation_np = np.float32(0.9)
    theta32 = tf.constant([theta_np], tf.float32)
    # reference adapter state layout: [B, N, 1]
    particles32 = tf.constant(
        h_np[None, :, None].astype(np.float32), tf.float32
    )
    reference_density = adapter.observation_value(
        theta32, particles32, tf.constant([observation_np]), 0
    )[0].numpy()
    mine = model.observation_log_density_fn(
        theta64,
        tf.constant(h_np[:, None], DTYPE),
        tf.constant([float(observation_np)], DTYPE),
    ).numpy()
    rel = np.max(np.abs(mine - reference_density) / np.maximum(np.abs(reference_density), 1.0))
    assert rel < 1.0e-4, f"KSC mixture density infidelity: rel {rel}"


def test_generalized_sv_reduction_slice_matches_kalman():
    """Reduction-slice gate (exact-math referent): at sigma_h -> 0 with
    initial h = 0 the model is linear-Gaussian in s (y = beta*s + N(0,1)),
    so the canonical pipeline value must match the 1-D Kalman likelihood
    within MC tolerance (declared: 0.05 abs, 2 seeds mean, N=4096)."""

    from bayesfilter.highdim.ledh_canonical_models_tf import (
        generalized_sv_canonical_model,
    )
    from bayesfilter.highdim.ledh_canonical_score_tf import (
        canonical_value_and_analytical_score,
    )

    rho_s, sigma_s, beta = 0.7, 0.9, 1.2
    theta_np = [
        float(np.arctanh(rho_s)),
        float(np.arctanh(0.5)),
        float(np.log(sigma_s)),
        float(np.log(1.0e-5)),  # sigma_h -> 0
        float(np.log(beta)),
    ]
    theta = tf.constant(theta_np, DTYPE)
    model, _sd = generalized_sv_canonical_model(theta)
    horizon, n = 3, 4096
    rng = np.random.default_rng(531)
    observations_np = rng.normal(0.0, 1.5, (horizon, 1))

    # 1-D Kalman on s: F=rho_s, Q=sigma_s^2, H=beta, R=exp(0)=1
    mean, var = 0.0, 1.0  # matches the particle initial cloud below
    kalman = 0.0
    for t in range(horizon):
        mean, var = rho_s * mean, rho_s**2 * var + sigma_s**2
        innovation_var = beta**2 * var + 1.0
        resid = observations_np[t, 0] - beta * mean
        kalman += -0.5 * (
            resid**2 / innovation_var + np.log(2.0 * np.pi * innovation_var)
        )
        gain = var * beta / innovation_var
        mean = mean + gain * resid
        var = (1.0 - gain * beta) * var

    values = []
    for seed in (0, 1):
        rng_p = np.random.default_rng(600 + seed)
        s0 = rng_p.standard_normal(n)
        initial = tf.constant(
            np.stack([s0, np.zeros(n)], axis=1), DTYPE
        )  # h = 0 exactly
        covs = tf.constant(
            np.stack([np.diag([1.0, 1.0e-10])] * n), DTYPE
        )
        noises = tf.constant(rng_p.standard_normal((horizon, n, 2)), DTYPE)
        value, _ = canonical_value_and_analytical_score(
            model, theta, initial, covs, noises,
            tf.constant(observations_np, DTYPE),
            substeps=8, with_score=False,
        )
        values.append(float(value.numpy()))
    err = abs(float(np.mean(values)) - kalman)
    assert err < 0.05, (
        f"reduction slice: canonical {np.mean(values):.4f} vs Kalman "
        f"{kalman:.4f} (err {err:.4f}, per-seed {values})"
    )
