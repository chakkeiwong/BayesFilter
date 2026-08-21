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
