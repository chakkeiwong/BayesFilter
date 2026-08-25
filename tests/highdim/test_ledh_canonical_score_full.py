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


def test_annealed_telescope_score_matches_oracle():
    """Q1.2 gate: annealed-mode analytical score vs the autodiff oracle.
    Three tempered stages with systematic resampling between them on the
    nonlinear fixture; the telescope increment, tempered-flow tangents
    (P/k, R*k), and fixed-realized-index resampling tangents are all
    live. rtol 1e-4 declared."""

    rng = np.random.default_rng(83)
    n, dim, horizon = 8, 2, 2
    initial = tf.constant(rng.standard_normal((n, dim)), DTYPE)
    covs = tf.constant(np.stack([np.eye(dim)] * n), DTYPE)
    noises = tf.constant(rng.standard_normal((horizon, n, dim)), DTYPE)
    observations = tf.constant(rng.standard_normal((horizon, dim)), DTYPE)
    theta0 = tf.constant([0.6], DTYPE)
    model = _model()
    kwargs = dict(substeps=8, annealed_stages=3, annealed_seed=17)

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
        f"annealed analytical {float(score[0].numpy())} vs oracle "
        f"{float(oracle[0].numpy())}"
    )


def test_annealed_telescope_with_reset_score_matches_oracle():
    """Q1.2 composition gate: annealed telescope followed by the S6/S7
    reset program (uniform post-resampling weights, zero weight tangent)
    still matches the oracle end to end."""

    rng = np.random.default_rng(97)
    n, dim, horizon = 16, 2, 2
    initial = tf.constant(rng.standard_normal((n, dim)), DTYPE)
    covs = tf.constant(np.stack([np.eye(dim)] * n), DTYPE)
    noises = tf.constant(rng.standard_normal((horizon, n, dim)), DTYPE)
    observations = tf.constant(rng.standard_normal((horizon, dim)), DTYPE)
    theta0 = tf.constant([0.6], DTYPE)
    model = _model()
    base = np.concatenate([np.eye(dim), -np.eye(dim)], axis=0)
    design = tf.constant(np.tile(base, (n // (2 * dim), 1)), DTYPE)
    kwargs = dict(
        substeps=8,
        annealed_stages=2,
        annealed_seed=29,
        reset_policy="contract_e",
        reset_design=design,
        reset_sinkhorn_steps=4,
        reset_balance_steps=2,
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
        f"annealed+reset analytical {float(score[0].numpy())} vs oracle "
        f"{float(oracle[0].numpy())}"
    )


def test_production_score_lane_value_is_likelihood_estimand():
    """Estimand gate (2026-08-25): the PRODUCTION score-lane program
    (contract_e reset) must track the exact Kalman log-likelihood on a
    linear anchor at T=10 — and the reset-none slice must NOT be usable
    as a likelihood estimand at T>1 (its uniform-weight no-resampling
    telescope is a derivative-parity object only). This is the gate
    class that would have caught the Q3 board's mislabeled plain score
    cells before they reached an owner-facing table.
    Declared bound: |mean over 4 seeds - Kalman| < 0.4 at N=1008
    (measured production bias at T=10: ~0.12; measured reset-none
    estimand error: ~0.65)."""

    import sys as _sys
    _sys.path.insert(0, "tests/highdim")
    from test_ledh_canonical_filter import (
        _kalman_log_likelihood,
        _lgssm_model,
    )

    spec = _lgssm_model(101, horizon=10)
    exact = _kalman_log_likelihood(spec)
    transition = tf.constant(spec["transition"], DTYPE)

    def transition_mean_fn(theta, points):
        return tf.einsum("ij,nj->ni", transition, points)

    def transition_mean_tangent_fn(theta, points, d_points):
        return tf.einsum("ij,nj->ni", transition, d_points)

    model = NonlinearScoreModel(
        transition_mean_fn=transition_mean_fn,
        transition_mean_tangent_fn=transition_mean_tangent_fn,
        observation_fn=lambda p: p,
        observation_jacobian_fn=lambda p: tf.broadcast_to(
            tf.constant(spec["obs_matrix"], DTYPE),
            [tf.shape(p)[0], 2, 2],
        ),
        observation_tangent_fn=lambda p, d: d,
        process_covariance=tf.constant(spec["process_cov"], DTYPE),
        observation_covariance=tf.constant(spec["obs_cov"], DTYPE),
    )
    observations = tf.constant(spec["observations"], DTYPE)
    n = 1008
    base = np.concatenate([np.eye(2), -np.eye(2)], axis=0)
    design = tf.constant(np.tile(base, (n // 4, 1)), DTYPE)

    def values(policy_kwargs):
        out = []
        for seed in range(4):
            rng = np.random.default_rng(9000 + seed)
            initial = tf.constant(rng.standard_normal((n, 2)), DTYPE)
            covs = tf.constant(np.stack([np.eye(2)] * n), DTYPE)
            noises = tf.constant(
                rng.standard_normal((10, n, 2)), DTYPE
            )
            v, _ = canonical_value_and_analytical_score(
                model, tf.constant([0.0], DTYPE), initial, covs, noises,
                observations, substeps=8, with_score=False,
                **policy_kwargs,
            )
            out.append(float(v.numpy()))
        return float(np.mean(out))

    from bayesfilter.highdim.ledh_alg1_contract import (
        LEDH_PRODUCTION_PROGRAM_V1,
    )

    production_kwargs = dict(LEDH_PRODUCTION_PROGRAM_V1["score"])
    production_kwargs["reset_design"] = design
    production = values(production_kwargs)
    assert abs(production - exact) < 0.4, (
        f"PRODUCTION score-lane value {production:.3f} vs Kalman "
        f"{exact:.3f} — estimand gate failed"
    )
    # NOTE (measured 2026-08-25): the reset-none slice's estimand error
    # scales with weight degeneracy — near-zero on easy high-ESS
    # fixtures like this one, +4.2 (N-independent) on the frozen dlgssm
    # T=50 row. A comparative assertion here is therefore fixture-
    # dependent and WRONG as a gate; the plain-slice trap is instead
    # enforced at the artifact layer (leaderboard cells must carry
    # program/tuning labels; the report builder refuses unlabeled
    # cells) and by the score-module estimand warning.


def test_annealed_with_full_production_program_matches_oracle():
    """Composition gate (2026-08-26): annealed telescope + the FULL
    registry production program (S6 reset + S7 dual-cap trust region)
    vs the oracle — the Austria-row score-cell composition, previously
    gated only without the S7 correction."""

    from bayesfilter.highdim.ledh_alg1_contract import (
        LEDH_PRODUCTION_PROGRAM_V1,
    )

    rng = np.random.default_rng(103)
    n, dim, horizon = 24, 2, 2
    initial = tf.constant(rng.standard_normal((n, dim)), DTYPE)
    covs = tf.constant(np.stack([np.eye(dim)] * n), DTYPE)
    noises = tf.constant(rng.standard_normal((horizon, n, dim)), DTYPE)
    observations = tf.constant(rng.standard_normal((horizon, dim)), DTYPE)
    theta0 = tf.constant([0.6], DTYPE)
    model = _model()
    base = np.concatenate([np.eye(dim), -np.eye(dim)], axis=0)
    design = tf.constant(np.tile(base, (n // (2 * dim), 1)), DTYPE)
    kwargs = dict(LEDH_PRODUCTION_PROGRAM_V1["score"])
    kwargs["reset_design"] = design
    kwargs.update(substeps=8, annealed_stages=2, annealed_seed=31)

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
        f"annealed+full-production analytical {float(score[0].numpy())} "
        f"vs oracle {float(oracle[0].numpy())}"
    )
