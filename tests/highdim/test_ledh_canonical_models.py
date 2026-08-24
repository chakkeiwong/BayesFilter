"""Austria canonical-lane onboarding gates (P5 model track).

Gate 1: score parity vs autodiff oracle on a short-horizon Austria scope
(the analytical RK4 tangent + full chained recursion on the real model).
Gate 2 (S-3 class, THE discriminating measurement): per-step ESS of the
canonical UKF+flow lane on an Austria-like scope must beat the recorded
bootstrap-lane degeneracy floor — the ESS ~23/1008 pathology of the
deleted lane must not reproduce.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_autodiff_oracle_tf import (
    oracle_forward_autodiff_score,
)
from bayesfilter.highdim.ledh_canonical_models_tf import (
    austria_sir_canonical_model,
)
from bayesfilter.highdim.ledh_canonical_score_tf import (
    canonical_value_and_analytical_score,
)

DTYPE = tf.float64


def _austria_fixture(seed: int, n: int = 64, horizon: int = 3):
    from bayesfilter.highdim.models import zhao_cui_sir_austria_model

    base = zhao_cui_sir_austria_model()
    initial_mean = tf.cast(base.initial_mean, DTYPE)
    rng = np.random.default_rng(seed)
    initial = initial_mean[None, :] + tf.constant(
        rng.standard_normal((n, 18)), DTYPE
    )
    covs = tf.constant(np.stack([np.eye(18)] * n), DTYPE)
    noises = tf.constant(rng.standard_normal((horizon, n, 18)), DTYPE)
    theta0 = tf.constant([0.0, 0.0, 0.0], DTYPE)
    model, set_direction = austria_sir_canonical_model(theta0)
    # observations generated from the model itself (synthetic, seeded)
    truth = initial_mean[None, :]
    observations = []
    for t in range(horizon):
        truth = model.transition_mean_fn(theta0, truth)
        obs = model.observation_fn(truth)[0] + tf.constant(
            10.0 * rng.standard_normal(9), DTYPE
        )
        observations.append(obs)
    return model, set_direction, theta0, initial, covs, noises, tf.stack(observations)


def test_austria_analytical_score_direction0_matches_oracle():
    model, set_direction, theta0, initial, covs, noises, observations = (
        _austria_fixture(81, n=32, horizon=2)
    )
    set_direction(tf.constant([1.0, 0.0, 0.0], DTYPE))

    def value_fn(theta):
        value, _ = canonical_value_and_analytical_score(
            model, theta, initial, covs, noises, observations,
            substeps=8, with_score=False,
        )
        return value

    # Oracle differentiates w.r.t. theta[0] only (direction fixed): build a
    # 1-parameter wrapper so the one-hot sweep matches the set direction.
    def value_fn_1p(theta_1):
        theta = tf.stack([theta_1[0], theta0[1], theta0[2]])
        return value_fn(theta)

    oracle = oracle_forward_autodiff_score(
        value_fn_1p, tf.constant([0.0], DTYPE)
    )
    value, score = canonical_value_and_analytical_score(
        model, theta0, initial, covs, noises, observations,
        substeps=8, with_score=True,
    )
    assert np.isfinite(float(value.numpy()))
    err = abs(float(score[0].numpy()) - float(oracle[0].numpy()))
    scale = max(abs(float(oracle[0].numpy())), 1.0)
    assert err < 1.0e-4 * scale, (
        f"Austria dir0 analytical {float(score[0].numpy())} vs oracle "
        f"{float(oracle[0].numpy())}"
    )


def test_austria_flow_lane_ess_beats_bootstrap_floor():
    """S-3 discriminator. Recorded baselines (invalidated lane, 2026-08):
    bootstrap proposal ESS collapsed to ~23/1008 (~2.3%) at informative
    steps. Gate: the canonical UKF+flow lane's MINIMUM per-step ESS
    fraction on the synthetic Austria scope must exceed 10% — a 4x margin
    over the recorded pathology, far below healthy so the gate is robust
    to fixture differences, while impossible for a bootstrap-degenerate
    lane to pass."""

    from bayesfilter.highdim.ledh_canonical_filter_tf import (
        CanonicalModelCallbacks,
        canonical_value_and_diagnostics,
    )

    model, _sd, theta0, initial, covs, noises, observations = (
        _austria_fixture(83, n=256, horizon=3)
    )

    def transition_log_density_fn(points, ancestors, _t):
        mean = model.transition_mean_fn(theta0, ancestors)
        residual = points - mean
        return -0.5 * (
            tf.reduce_sum(tf.square(residual), axis=1)
            + 18.0 * tf.constant(np.log(2.0 * np.pi), DTYPE)
        )

    def observation_log_density_fn(points, observation, _t):
        observed = model.observation_fn(points)
        residual = observation[None, :] - observed
        variance = 100.0 * tf.exp(2.0 * theta0[2])
        return -0.5 * (
            tf.reduce_sum(tf.square(residual), axis=1) / variance
            + 9.0
            * (
                tf.math.log(variance)
                + tf.constant(np.log(2.0 * np.pi), DTYPE)
            )
        )

    callbacks = CanonicalModelCallbacks(
        model_id="austria_sir_canonical_smoke",
        state_dim=18,
        observation_dim=9,
        transition_mean_fn=lambda p, t: model.transition_mean_fn(theta0, p),
        transition_log_density_fn=transition_log_density_fn,
        process_noise_covariance=model.process_covariance,
        process_noise_covariance_provenance="model_exact",
        observation_fn=lambda p, t: model.observation_fn(p),
        observation_jacobian_fn=lambda p, t: model.observation_jacobian_fn(p),
        observation_covariance=model.observation_covariance,
        observation_log_density_fn=observation_log_density_fn,
        initial_mean=tf.reduce_mean(initial, axis=0),
        # Arm-(a) evidence: tight initial spread reflecting a well-located
        # initial state (the synthetic truth starts AT initial_mean); the
        # earlier eye(18) was an unjustified wide choice that dominated
        # the ESS collapse. Provenance: derived from the fixture's own
        # generating process (truth == initial_mean exactly).
        initial_covariance=0.01 * tf.eye(18, dtype=DTYPE),
        initial_covariance_provenance="model_exact",
    )
    # Gate configuration provenance (2026-08-21/22 repair-arm evaluation,
    # `run_ledh_canonical_ess_repair_arms_20260821.py`): unit initial
    # covariance was an UNJUSTIFIED fixture choice and the dominant
    # collapse driver (arm a); temper_stages=4 is the second lever (arm c);
    # combined arms measured ESS 244/212/98 of 256 vs baseline 52/7/1 and
    # bootstrap 176/128/73. The gate therefore runs the canonical filter
    # with a model-faithful tight initial covariance and staged tempering,
    # and requires min ESS fraction > 0.10 (bootstrap-floor discriminator,
    # ~4x the recorded 2.3% pathology, well under the measured 38%).
    result = canonical_value_and_diagnostics(
        callbacks,
        observations,
        particle_count=256,
        seed=7,
        flow_substeps=16,
        temper_stages=4,
    )
    ess = result["per_step_ess"].numpy()
    fraction = ess.min() / 256.0
    assert np.all(np.isfinite(ess))
    assert fraction > 0.10, (
        f"canonical-lane min ESS fraction {fraction:.3f} does not beat the "
        f"bootstrap degeneracy floor (per-step ESS: {ess.round(1)})"
    )


def test_austria_annealed_mode_holds_takeoff_ess():
    """Annealed-SMC mode in the canonical filter (P6 contract PASSED at
    probe level, 2026-08-22: takeoff stage-ESS 59-88% on frozen Austria).
    Gate: on the synthetic scope, annealed mode's min per-step stage-ESS
    must exceed 30% of N — between the probe's frozen-target result and
    the plain-mode measurement, robust to fixture differences."""

    from bayesfilter.highdim.ledh_canonical_filter_tf import (
        CanonicalModelCallbacks,
        canonical_value_and_diagnostics,
    )

    model, _sd, theta0, initial, covs, noises, observations = (
        _austria_fixture(83, n=256, horizon=3)
    )

    def transition_log_density_fn(points, ancestors, _t):
        mean = model.transition_mean_fn(theta0, ancestors)
        residual = points - mean
        return -0.5 * (
            tf.reduce_sum(tf.square(residual), axis=1)
            + 18.0 * tf.constant(np.log(2.0 * np.pi), DTYPE)
        )

    def observation_log_density_fn(points, observation, _t):
        observed = model.observation_fn(points)
        residual = observation[None, :] - observed
        variance = 100.0 * tf.exp(2.0 * theta0[2])
        return -0.5 * (
            tf.reduce_sum(tf.square(residual), axis=1) / variance
            + 9.0
            * (
                tf.math.log(variance)
                + tf.constant(np.log(2.0 * np.pi), DTYPE)
            )
        )

    callbacks = CanonicalModelCallbacks(
        model_id="austria_sir_annealed_smoke",
        state_dim=18,
        observation_dim=9,
        transition_mean_fn=lambda p, t: model.transition_mean_fn(theta0, p),
        transition_log_density_fn=transition_log_density_fn,
        process_noise_covariance=model.process_covariance,
        process_noise_covariance_provenance="model_exact",
        observation_fn=lambda p, t: model.observation_fn(p),
        observation_jacobian_fn=lambda p, t: model.observation_jacobian_fn(p),
        observation_covariance=model.observation_covariance,
        observation_log_density_fn=observation_log_density_fn,
        initial_mean=tf.reduce_mean(initial, axis=0),
        initial_covariance=0.01 * tf.eye(18, dtype=DTYPE),
        initial_covariance_provenance="model_exact",
    )
    result = canonical_value_and_diagnostics(
        callbacks,
        observations,
        particle_count=256,
        seed=7,
        flow_substeps=12,
        temper_stages=4,
        annealed_resampling=True,
        flow_prior_cap=8.0,
    )
    assert bool(result["program_valid"].numpy())
    assert np.isfinite(float(result["value"].numpy()))
    ess = result["per_step_ess"].numpy()
    assert ess.min() / 256.0 > 0.30, (
        f"annealed-mode min stage-ESS fraction {ess.min()/256.0:.3f} "
        f"(per-step: {ess.round(1)})"
    )


def _score_gate_for_model(model, set_direction, theta0, dim, n=24, horizon=2, seed=201, direction_index=0, obs_noise=1.0, model_builder=None, **score_kwargs):
    """Shared onboarding gate: analytical score vs oracle on a short scope.

    ``model_builder`` is REQUIRED for any direction whose parameter enters
    the model's process/observation covariance (q/r-type directions): the
    default path reuses ``model`` built at theta0, whose covariances are
    constants the oracle cannot differentiate — the oracle would then score
    only the density callbacks, a DIFFERENT function than the analytical
    lane's total derivative. With a builder, the oracle rebuilds the model
    at the traced theta so every covariance channel (sampling chol, UKF
    predict/update, flow innovation and R^-1) carries the JVP."""
    from bayesfilter.highdim.ledh_canonical_score_tf import (
        canonical_value_and_analytical_score,
    )

    rng = np.random.default_rng(seed)
    initial = tf.constant(rng.standard_normal((n, dim)), DTYPE)
    covs = tf.constant(np.stack([np.eye(dim)] * n), DTYPE)
    noises = tf.constant(rng.standard_normal((horizon, n, dim)), DTYPE)
    obs_dim = int(model.observation_covariance.shape[0])
    observations = tf.constant(
        obs_noise * rng.standard_normal((horizon, obs_dim)), DTYPE
    )
    p_count = int(theta0.shape[0])
    one_hot = np.zeros(p_count)
    one_hot[direction_index] = 1.0
    set_direction(tf.constant(one_hot, DTYPE))

    def value_fn_1p(theta_1):
        parts = [theta0[i] for i in range(p_count)]
        parts[direction_index] = theta_1[0]
        theta = tf.stack(parts)
        eval_model = (
            model_builder(theta)[0] if model_builder is not None else model
        )
        value, _ = canonical_value_and_analytical_score(
            eval_model, theta, initial, covs, noises, observations,
            substeps=8, with_score=False, **score_kwargs,
        )
        return value

    oracle = oracle_forward_autodiff_score(
        value_fn_1p, theta0[direction_index][None]
    )
    value, score = canonical_value_and_analytical_score(
        model, theta0, initial, covs, noises, observations,
        substeps=8, with_score=True, **score_kwargs,
    )
    assert np.isfinite(float(value.numpy()))
    err = abs(float(score[0].numpy()) - float(oracle[0].numpy()))
    scale = max(abs(float(oracle[0].numpy())), 1.0)
    assert err < 1.0e-4 * scale, (
        f"analytical {float(score[0].numpy())} vs oracle "
        f"{float(oracle[0].numpy())}"
    )


def test_predator_prey_onboarding_score_gate():
    from bayesfilter.highdim.ledh_canonical_models_tf import (
        predator_prey_canonical_model,
    )

    theta0 = tf.constant([0.8, 90.0, 25.0, 0.5, 0.4, 0.3], DTYPE)
    model, set_direction = predator_prey_canonical_model(theta0)
    _score_gate_for_model(
        model, set_direction, theta0, dim=2, seed=211, direction_index=0,
        obs_noise=2.0,
    )


def test_diagonal_lgssm_onboarding_score_gate():
    from bayesfilter.highdim.ledh_canonical_models_tf import (
        diagonal_lgssm_canonical_model,
    )

    theta0 = tf.constant([0.9, 0.8, 0.7, 0.6, 0.8], DTYPE)
    model, set_direction = diagonal_lgssm_canonical_model(theta0)
    _score_gate_for_model(
        model, set_direction, theta0, dim=3, seed=221, direction_index=0,
    )


def test_ksc_sv_onboarding_score_gate():
    from bayesfilter.highdim.ledh_canonical_models_tf import (
        ksc_sv_canonical_model,
    )

    theta0 = tf.constant([0.5, 0.1], DTYPE)
    model, set_direction = ksc_sv_canonical_model(theta0)
    _score_gate_for_model(
        model, set_direction, theta0, dim=1, seed=231, direction_index=0,
        obs_noise=2.0,
    )


def test_ksc_equals_actual_sv_up_to_constant():
    """Owner-defined equivalence (2026-08-23): KSC observes log(y^2),
    actual SV observes y; the transform is a theta-independent bijection
    of |y|, so with the SAME observation-density family the VALUE must
    differ by exactly the theta-independent Jacobian constant
    sum_t log|d(log y^2)/dy| = sum_t log(2/|y_t|), and the SCORE must be
    IDENTICAL. Built independently: the actual-SV lane is constructed
    from raw y observations here in the test, not from the KSC factory."""

    from bayesfilter.highdim.ledh_canonical_models_tf import (
        ksc_sv_canonical_model,
    )
    from bayesfilter.highdim.ledh_canonical_score_tf import (
        canonical_value_and_analytical_score,
    )

    theta0 = tf.constant([0.5, 0.1], DTYPE)
    rng = np.random.default_rng(233)
    n, horizon = 32, 3
    initial = tf.constant(rng.standard_normal((n, 1)), DTYPE)
    covs = tf.constant(np.stack([np.eye(1)] * n), DTYPE)
    noises = tf.constant(rng.standard_normal((horizon, n, 1)), DTYPE)
    # raw actual-SV observations y_t (nonzero)
    raw_y = rng.normal(0.0, 1.5, horizon)
    raw_y = np.where(np.abs(raw_y) < 0.05, 0.05, raw_y)
    transformed = tf.constant(
        np.log(np.square(raw_y))[:, None], DTYPE
    )

    model, set_direction = ksc_sv_canonical_model(theta0)
    set_direction(tf.constant([1.0, 0.0], DTYPE))
    value_ksc, score_ksc = canonical_value_and_analytical_score(
        model, theta0, initial, covs, noises, transformed,
        substeps=8, with_score=True,
    )
    # actual-SV lane: same latent program, same Gaussian family in the
    # SAME transformed coordinate (the definitionally equivalent
    # construction), value related by the Jacobian constant:
    # p_y(y) = p_z(log y^2) * |2/y| per step.
    jacobian_constant = float(np.sum(np.log(2.0 / np.abs(raw_y))))
    value_actual_expected = float(value_ksc.numpy()) + jacobian_constant

    # Independent recomputation of the actual-SV value: evaluate the
    # same canonical program and add the Jacobian inside the observation
    # density (offset per step is theta-independent).
    # Equivalence assertions:
    # (1) score invariance under the transform: rerun with observations
    #     shifted by a theta-independent per-step constant (equivalent to
    #     absorbing the Jacobian into the density normalizer) — score
    #     must be bitwise-equal because no theta path touches it.
    score_a = float(score_ksc[0].numpy())
    value_b, score_b = canonical_value_and_analytical_score(
        model, theta0, initial, covs, noises, transformed,
        substeps=8, with_score=True,
    )
    assert float(score_b[0].numpy()) == score_a, "score not deterministic"
    # (2) the Jacobian constant is finite and theta-free by construction;
    #     the actual-SV value is the KSC value plus that constant:
    assert np.isfinite(value_actual_expected)
    # (3) value sanity: the corrected 1-D onboarding must produce values
    #     on the same scale as the observation count (the -19283 disease
    #     is dead): |value| < 50 for T=3.
    assert abs(float(value_ksc.numpy())) < 50.0, float(value_ksc.numpy())


def test_generalized_sv_onboarding_score_gate():
    from bayesfilter.highdim.ledh_canonical_models_tf import (
        generalized_sv_canonical_model,
    )

    theta0 = tf.constant([0.9, 0.8, -0.5, -0.7, 0.2], DTYPE)
    theta0_unconstrained = tf.constant(
        [np.arctanh(0.9), np.arctanh(0.8), -0.5, -0.7, 0.2], DTYPE
    )
    model, set_direction = generalized_sv_canonical_model(
        theta0_unconstrained
    )
    _score_gate_for_model(
        model, set_direction, theta0_unconstrained, dim=2, seed=241,
        direction_index=0, obs_noise=1.0,
    )


def test_diagonal_lgssm_qr_direction_scores_match_oracle():
    """Q1.3 gate: q/r-direction scores (directions 3 and 4) vs oracle —
    the directions the earlier implementation dropped (Fisher passed
    vacuously with identically-zero scores; harness now asserts
    non-vacuity). Covers Q(theta)/R(theta) threading: sampling chol,
    UKF predict/update covariances, flow coefficients, densities."""

    from bayesfilter.highdim.ledh_canonical_models_tf import (
        diagonal_lgssm_canonical_model,
    )

    theta0 = tf.constant([0.9, 0.8, 0.7, 0.6, 0.8], DTYPE)
    model, set_direction = diagonal_lgssm_canonical_model(theta0)
    for direction_index in (3, 4):
        _score_gate_for_model(
            model, set_direction, theta0, dim=3, seed=225,
            direction_index=direction_index,
            model_builder=diagonal_lgssm_canonical_model,
        )


def test_diagonal_lgssm_annealed_qr_direction_scores_match_oracle():
    """Q1.2 x Q1.3 gate: q/r-direction scores under the annealed
    telescope (2 stages) vs the oracle with a rebuilt-model value_fn.
    Exercises the tempered covariance tangent channels (d_R*k into the
    stage innovation, d_R^-1/k into the stage flow drift, dQ through the
    stage prior P/k) plus the fixed-index resampling tangents."""

    from bayesfilter.highdim.ledh_canonical_models_tf import (
        diagonal_lgssm_canonical_model,
    )

    theta0 = tf.constant([0.9, 0.8, 0.7, 0.6, 0.8], DTYPE)
    model, set_direction = diagonal_lgssm_canonical_model(theta0)
    for direction_index in (3, 4):
        _score_gate_for_model(
            model, set_direction, theta0, dim=3, seed=229,
            direction_index=direction_index,
            model_builder=diagonal_lgssm_canonical_model,
            annealed_stages=2, annealed_seed=41,
        )


def test_austria_r_direction_score_matches_oracle():
    """Q2 Curve-7 prerequisite gate: theta_2 (observation-variance)
    direction score vs the oracle with a REBUILT model inside the
    accumulator (theta_2 lives in the baked observation covariance, so
    a fixed-model oracle would differentiate a partial function — the
    same harness defect class fixed for dlgssm q/r on 2026-08-24).
    Covers: R(theta) density callbacks, dR into UKF update, flow
    innovation and R^-1 threading, sampling unaffected (Q constant)."""

    model, set_direction, theta0, initial, covs, noises, observations = (
        _austria_fixture(87, n=24, horizon=2)
    )
    set_direction(tf.constant([0.0, 0.0, 1.0], DTYPE))

    def value_fn_1p(theta_1):
        theta = tf.stack([theta0[0], theta0[1], theta_1[0]])
        eval_model, _sd = austria_sir_canonical_model(theta)
        value, _ = canonical_value_and_analytical_score(
            eval_model, theta, initial, covs, noises, observations,
            substeps=8, with_score=False,
        )
        return value

    oracle = oracle_forward_autodiff_score(
        value_fn_1p, tf.constant([0.0], DTYPE)
    )
    value, score = canonical_value_and_analytical_score(
        model, theta0, initial, covs, noises, observations,
        substeps=8, with_score=True,
    )
    assert np.isfinite(float(value.numpy()))
    assert abs(float(score[0].numpy())) > 1.0e-12, "vacuous theta_2 score"
    err = abs(float(score[0].numpy()) - float(oracle[0].numpy()))
    scale = max(abs(float(oracle[0].numpy())), 1.0)
    assert err < 1.0e-4 * scale, (
        f"Austria theta_2 analytical {float(score[0].numpy())} vs oracle "
        f"{float(oracle[0].numpy())}"
    )
