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


def test_austria_reduction_slice_matches_kalman():
    """Q1.4 reduction-slice gate (exact-math referent): theta_0 = -20
    makes kappa ~ 2e-10, the infection term vanishes, and the RK4
    dynamics become LINEAR (diffusion + nu decay). The propagator is
    extracted by pushing basis vectors through the model's own
    transition_mean_fn; exact 18-dim Kalman then anchors the canonical
    pipeline value. Declared: |mean over 2 seeds - Kalman| < 0.1."""

    from bayesfilter.highdim.ledh_canonical_models_tf import (
        austria_sir_canonical_model,
    )
    from bayesfilter.highdim.ledh_canonical_filter_tf import (
        CanonicalModelCallbacks,
        canonical_value_and_diagnostics,
    )
    from bayesfilter.highdim.models import zhao_cui_sir_austria_model

    theta = tf.constant([-20.0, 0.0, 0.0], DTYPE)
    model, _sd = austria_sir_canonical_model(theta)

    # linearity check + propagator extraction
    basis = tf.constant(np.eye(18), DTYPE)
    pushed_basis = model.transition_mean_fn(theta, basis).numpy()
    pushed_zero = model.transition_mean_fn(
        theta, tf.zeros([1, 18], DTYPE)
    ).numpy()
    assert np.max(np.abs(pushed_zero)) < 1e-8, "dynamics not homogeneous"
    transition = pushed_basis.T  # F e_j = column j
    probe = np.array([0.3, -0.2] * 9)
    linear_err = np.max(
        np.abs(
            model.transition_mean_fn(
                theta, tf.constant(probe[None, :], DTYPE)
            ).numpy()[0]
            - transition @ probe
        )
    )
    assert linear_err < 1e-8, f"dynamics not linear at kappa=0: {linear_err}"

    obs_matrix = np.zeros((9, 18))
    for o in range(9):
        obs_matrix[o, 2 * o + 1] = 1.0
    variance = 100.0
    horizon = 2
    rng = np.random.default_rng(541)
    initial_mean_np = zhao_cui_sir_austria_model().initial_mean.numpy()
    observations_np = (
        obs_matrix @ initial_mean_np
    )[None, :] + 5.0 * rng.standard_normal((horizon, 9))

    # exact Kalman
    mean, cov = initial_mean_np.copy(), np.eye(18)
    kalman = 0.0
    for t in range(horizon):
        mean = transition @ mean
        cov = transition @ cov @ transition.T + np.eye(18)
        s = obs_matrix @ cov @ obs_matrix.T + variance * np.eye(9)
        resid = observations_np[t] - obs_matrix @ mean
        sign, logdet = np.linalg.slogdet(2.0 * np.pi * s)
        kalman += -0.5 * (resid @ np.linalg.solve(s, resid) + logdet)
        gain = cov @ obs_matrix.T @ np.linalg.inv(s)
        mean = mean + gain @ resid
        cov = (np.eye(18) - gain @ obs_matrix) @ cov

    def transition_log_density_fn(points, ancestors, _t):
        m = model.transition_mean_fn(theta, ancestors)
        r = points - m
        return -0.5 * (
            tf.reduce_sum(tf.square(r), axis=1)
            + 18.0 * tf.constant(np.log(2.0 * np.pi), DTYPE)
        )

    def observation_log_density_fn(points, observation, _t):
        observed = model.observation_fn(points)
        r = observation[None, :] - observed
        return -0.5 * (
            tf.reduce_sum(tf.square(r), axis=1) / variance
            + 9.0 * tf.constant(np.log(2.0 * np.pi * variance), DTYPE)
        )

    callbacks = CanonicalModelCallbacks(
        model_id="austria_reduction_slice",
        state_dim=18,
        observation_dim=9,
        transition_mean_fn=lambda p, t: model.transition_mean_fn(theta, p),
        transition_log_density_fn=transition_log_density_fn,
        process_noise_covariance=tf.eye(18, dtype=DTYPE),
        process_noise_covariance_provenance="model_exact",
        observation_fn=lambda p, t: model.observation_fn(p),
        observation_jacobian_fn=lambda p, t: model.observation_jacobian_fn(p),
        observation_covariance=100.0 * tf.eye(9, dtype=DTYPE),
        observation_log_density_fn=observation_log_density_fn,
        initial_mean=tf.constant(initial_mean_np, DTYPE),
        initial_covariance=tf.eye(18, dtype=DTYPE),
        initial_covariance_provenance="model_exact",
    )
    values = []
    for seed in (0, 1):
        result = canonical_value_and_diagnostics(
            callbacks,
            tf.constant(observations_np, DTYPE),
            particle_count=4096,
            seed=seed,
            flow_substeps=12,
        )
        assert bool(result["program_valid"].numpy())
        values.append(float(result["value"].numpy()))
    err = abs(float(np.mean(values)) - kalman)
    assert err < 0.1, (
        f"Austria reduction: canonical {np.mean(values):.4f} vs Kalman "
        f"{kalman:.4f} (err {err:.4f}, per-seed {values})"
    )



def test_bootstrap_comparator_matches_kalman_on_linear_anchor():
    """Fidelity item #6 gate (2026-08-25): the Q3 bootstrap comparator
    (per-step systematic resampling) must converge to the exact Kalman
    value on the linear anchor — the class of gate that would have
    caught the slice-2 comparator's no-resampling defect (which treated
    incoming weights as uniform while never resampling; wrong relative
    to the bootstrap-PF likelihood decomposition for T > 1).
    Declared: |mean over 5 seeds - Kalman| < 3*seed-SE + 0.5 nats at
    N=4096, T=8."""

    import sys as _sys
    _sys.path.insert(0, "docs/benchmarks")
    from run_q3_leaderboard_20260824 import (
        bootstrap_value,
        _gaussian_obs_log,
    )
    from test_ledh_canonical_filter import (
        _kalman_log_likelihood,
        _lgssm_model,
    )

    spec = _lgssm_model(303, horizon=8)
    exact = _kalman_log_likelihood(spec)
    transition = tf.constant(spec["transition"], DTYPE)

    class _LG:
        transition_mean_fn = staticmethod(
            lambda theta, p: tf.einsum("ij,nj->ni", transition, p)
        )
        observation_fn = staticmethod(lambda p: p)
        process_covariance = tf.constant(spec["process_cov"], DTYPE)
        observation_covariance = tf.constant(spec["obs_cov"], DTYPE)
        observation_log_density_fn = None

    obs_log = _gaussian_obs_log(_LG, None)
    observations = tf.constant(spec["observations"], DTYPE)
    initial_mean = tf.constant(spec["initial_mean"], DTYPE)
    values = [
        bootstrap_value(_LG, None, observations, obs_log, initial_mean,
                        2, 4096, seed)
        for seed in range(5)
    ]
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(5))
    assert abs(mean - exact) < 3.0 * se + 0.5, (
        f"bootstrap comparator {mean:.3f} vs Kalman {exact:.3f} "
        f"(SE {se:.3f}, per-seed {values})"
    )
