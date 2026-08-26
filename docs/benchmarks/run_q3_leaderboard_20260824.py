"""Q3 six-model leaderboard runner (per-row processes, G-5 stamped).

Contract: `docs/plans/bayesfilter-q3-leaderboard-execution-plan-2026-08-24.md`.
Rows: linear-2d fixture (exact anchor), diagonal LGSSM (frozen T=50,
exact Kalman value AND score references), predator-prey (frozen T=20),
KSC SV (frozen T=1000), generalized SV (simulated T=20), Austria SIR
(frozen T=20; Q2-calibrated annealed arms). Value arms per row:
canonical filter, bootstrap PF (systematic resampling), UKF Gaussian
filter, exact reference where linear. Score cells: analytical (8 seeds)
+ single-seed central-FD self-consistency on the score-lane value.

G-5: the runner refuses to run unless the canonical gate battery passes
at the current commit; artifacts carry `alg1_conformance`.

Per-scope discipline: the Q2 calibration (k=4, c=8, f32/TF32+cap) is
applied ONLY to the Austria row; other rows run plain canonical f64
(their onboarding-gate class). UKF-GF is a Gaussian-approximation
comparator and is density-misspecified on the mixture/heteroskedastic
rows (recorded per row).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tests", "highdim"))

import numpy as np  # noqa: E402  (diagnostics/simulation)
import tensorflow as tf  # noqa: E402

# Memory growth MUST be set before any bayesfilter import (module-level
# tf constants initialize the GPU); the registry import below would
# otherwise lock the devices and the runner would fail closed.
for _gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_gpu, True)

DTYPE = tf.float64
VALUE_SEEDS = list(range(16))
SCORE_SEEDS = list(range(8))


# ---------------------------------------------------------------- helpers
def _gaussian_obs_log(model, theta):
    r = model.observation_covariance
    chol = tf.linalg.cholesky(r)
    obs_dim = int(r.shape[0])
    logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
    norm = obs_dim * tf.constant(np.log(2.0 * np.pi), DTYPE) + logdet

    def fn(points, obs, _t):
        observed = model.observation_fn(points)
        residual = obs[None, :] - observed
        solved = tf.linalg.triangular_solve(
            chol, tf.transpose(residual)
        )
        quad = tf.reduce_sum(tf.square(solved), axis=0)
        return -0.5 * (quad + norm)

    return fn


def _density_obs_log(model, theta):
    if model.observation_log_density_fn is not None:
        return lambda points, obs, _t: model.observation_log_density_fn(
            theta, points, obs
        )
    return _gaussian_obs_log(model, theta)


def _gaussian_trans_log(model, theta):
    q = model.process_covariance
    chol = tf.linalg.cholesky(q)
    dim = int(q.shape[0])
    logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
    norm = dim * tf.constant(np.log(2.0 * np.pi), DTYPE) + logdet

    def fn(points, ancestors, _t):
        mean = model.transition_mean_fn(theta, ancestors)
        residual = points - mean
        solved = tf.linalg.triangular_solve(
            chol, tf.transpose(residual)
        )
        quad = tf.reduce_sum(tf.square(solved), axis=0)
        return -0.5 * (quad + norm)

    return fn


def make_callbacks(model, theta, model_id, dim, obs_dim, initial_mean):
    from bayesfilter.highdim.ledh_canonical_filter_tf import (
        CanonicalModelCallbacks,
    )

    return CanonicalModelCallbacks(
        model_id=model_id,
        state_dim=dim,
        observation_dim=obs_dim,
        transition_mean_fn=lambda p, t: model.transition_mean_fn(theta, p),
        transition_log_density_fn=_gaussian_trans_log(model, theta),
        process_noise_covariance=model.process_covariance,
        process_noise_covariance_provenance="model_exact",
        observation_fn=lambda p, t: model.observation_fn(p),
        observation_jacobian_fn=lambda p, t: model.observation_jacobian_fn(p),
        observation_covariance=model.observation_covariance,
        observation_log_density_fn=_density_obs_log(model, theta),
        initial_mean=initial_mean,
        initial_covariance=tf.eye(dim, dtype=DTYPE),
        initial_covariance_provenance="model_exact",
    )


def bootstrap_value(model, theta, observations, obs_log_fn, initial_mean,
                    dim, n, seed):
    """Bootstrap PF with per-step SYSTEMATIC resampling (fidelity item
    #6: the slice-2 comparator propagated without resampling while
    treating incoming weights as uniform — wrong for T > 1)."""

    rng = np.random.default_rng(seed)
    states = tf.constant(
        initial_mean.numpy()[None, :] + rng.standard_normal((n, dim)),
        DTYPE,
    )
    chol = tf.linalg.cholesky(model.process_covariance)
    total = 0.0
    for t in range(int(observations.shape[0])):
        obs = observations[t]
        anchors = model.transition_mean_fn(theta, states)
        noise = tf.constant(rng.standard_normal((n, dim)), DTYPE)
        children = anchors + tf.einsum("ij,nj->ni", chol, noise)
        ol = obs_log_fn(children, obs, t)
        total += float(tf.reduce_logsumexp(ol - np.log(n)).numpy())
        w = tf.exp(ol - tf.reduce_logsumexp(ol)).numpy()
        positions = (rng.uniform() + np.arange(n)) / n
        cumulative = np.cumsum(w)
        cumulative[-1] = 1.0
        idx = np.searchsorted(cumulative, positions)
        states = tf.gather(children, tf.constant(idx, tf.int32))
    return total


def ukf_gaussian_filter_value(model, theta, initial_mean, initial_cov,
                              observations):
    from bayesfilter.highdim.ledh_ukf_lifecycle_tf import (
        ukf_predict_per_particle,
        ukf_update_per_particle,
    )

    mean = initial_mean[None, :]
    cov = initial_cov[None, :, :]
    total = 0.0
    for t in range(int(observations.shape[0])):
        obs = observations[t]
        pm, pc = ukf_predict_per_particle(
            mean, cov, lambda p: model.transition_mean_fn(theta, p),
            model.process_covariance,
        )
        observed_pred = model.observation_fn(pm)[0]
        h = model.observation_jacobian_fn(pm)[0]
        innovation_cov = (
            tf.einsum("oi,ij,pj->op", h, pc[0], h)
            + model.observation_covariance
        )
        residual = (obs - observed_pred).numpy()
        sign, logdet = np.linalg.slogdet(
            2.0 * np.pi * innovation_cov.numpy()
        )
        total += -0.5 * (
            residual @ np.linalg.solve(innovation_cov.numpy(), residual)
            + logdet
        )
        mean, cov = ukf_update_per_particle(
            pm, pc, lambda p: model.observation_fn(p),
            model.observation_covariance, obs,
        )
    return float(total)


def canonical_filter_values(callbacks, observations, seeds, **filter_kwargs):
    from bayesfilter.highdim.ledh_canonical_filter_tf import (
        canonical_value_and_diagnostics,
    )

    values, valids, min_ess = [], [], []
    for seed in seeds:
        result = canonical_value_and_diagnostics(
            callbacks, observations, particle_count=1008, seed=seed,
            flow_substeps=16, resample_seed=seed,
            **{**PRODUCTION_FILTER_KWARGS, **filter_kwargs},
        )
        values.append(float(result["value"].numpy()))
        valids.append(bool(result["program_valid"].numpy()))
        min_ess.append(float(result["per_step_ess"].numpy().min()))
    return values, valids, min_ess


from bayesfilter.highdim.ledh_alg1_contract import (  # noqa: E402
    LEDH_PRODUCTION_PROGRAM_V1,
)

# Value-filter production kwargs FROM THE REGISTRY (owner directive
# 2026-08-26: dual-cap trust region is a REQUIRED mechanism — the
# covariance-explosion control; omitting it is a labeled deviation,
# never a default).
PRODUCTION_FILTER_KWARGS = dict(LEDH_PRODUCTION_PROGRAM_V1["filter"])


def production_score_kwargs(dim, n=1008):
    """Production score-lane config FROM THE REGISTRY (S6 contract_e
    reset + S7 dual-cap trust-region correction) plus the scope-shaped
    tiled +/- unit design."""
    base = np.concatenate([np.eye(dim), -np.eye(dim)], axis=0)
    design = tf.constant(np.tile(base, (n // (2 * dim), 1)), DTYPE)
    kwargs = dict(LEDH_PRODUCTION_PROGRAM_V1["score"])
    kwargs["reset_design"] = design
    return kwargs


UNTUNED = ("UNTUNED (no per-scope tuning artifact; registry warm-start "
           "values: trust radius 0.5 warm start per Q2 Curve-2, "
           "epsilon 2.0 warm start per gap A2)")


def score_cells(model, set_direction, theta, dim, observations,
                direction_index, seeds, annealed_stages=1,
                initial_mean=None, self_consistency="fd",
                reset_kwargs=None, program="UNLABELED",
                tuning=UNTUNED):
    from bayesfilter.highdim.ledh_canonical_score_tf import (
        canonical_value_and_analytical_score,
    )

    p_count = int(theta.shape[0])
    one_hot = np.zeros(p_count)
    one_hot[direction_index] = 1.0
    set_direction(tf.constant(one_hot, DTYPE))
    horizon = int(observations.shape[0])
    n = 1008
    mean_np = (
        np.zeros(dim) if initial_mean is None else initial_mean.numpy()
    )

    def run(theta_v, seed, with_score):
        rng = np.random.default_rng(9000 + seed)
        initial = tf.constant(
            mean_np[None, :] + rng.standard_normal((n, dim)), DTYPE
        )
        covs = tf.constant(np.stack([np.eye(dim)] * n), DTYPE)
        noises = tf.constant(
            rng.standard_normal((horizon, n, dim)), DTYPE
        )
        return canonical_value_and_analytical_score(
            model, theta_v, initial, covs, noises, observations,
            substeps=8, with_score=with_score,
            annealed_stages=annealed_stages, annealed_seed=17,
            **(reset_kwargs or {}),
        )

    scores = []
    crashed = 0
    for seed in seeds:
        try:
            _v, s = run(theta, seed, True)
            scores.append(float(s[0].numpy()))
        except Exception as error:  # crash -> recorded veto (2026-08-26)
            print(f"[score-cell VETO] seed={seed}: {error}", flush=True)
            scores.append(float("nan"))
            crashed += 1
    analytical_seed0 = scores[0]
    if crashed == len(seeds):
        return {
            "program": program,
            "tuning": tuning,
            "crashed_seeds": crashed,
            "all_finite": False,
            "direction_index": direction_index,
            "analytical_scores": scores,
            "mean": float("nan"),
            "seed_spread": float("nan"),
            "self_consistency_reference": float("nan"),
            "self_consistency_kind": "unavailable (all seeds vetoed)",
            "self_consistency_rel_err_seed0": float("nan"),
            "annealed_stages": annealed_stages,
            "note": "HARD VETO: every seed crashed nonfinite",
        }
    try:
        return _self_consistency_block(
            self_consistency, run, theta, seeds, scores, direction_index,
            p_count, program, tuning, crashed, annealed_stages,
            analytical_seed0,
        )
    except Exception as error:  # reference crash -> recorded (2026-08-26)
        print(f"[score-reference VETO] {error}", flush=True)
        return {
            "program": program,
            "tuning": tuning,
            "crashed_seeds": crashed,
            "all_finite": bool(np.all(np.isfinite(scores))),
            "direction_index": direction_index,
            "analytical_scores": scores,
            "mean": float(np.nanmean(scores)),
            "seed_spread": float(np.nanstd(scores)),
            "self_consistency_reference": float("nan"),
            "self_consistency_kind": "unavailable (reference crashed)",
            "self_consistency_rel_err_seed0": float("nan"),
            "annealed_stages": annealed_stages,
            "note": "reference computation crashed nonfinite (recorded)",
        }


def _self_consistency_block(self_consistency, run, theta, seeds, scores,
                            direction_index, p_count, program, tuning,
                            crashed, annealed_stages, analytical_seed0):
    if self_consistency == "oracle":
        # Annealed rows: central FD is invalid across resampling-boundary
        # crossings; the forward-autodiff oracle differentiates the SAME
        # fixed-realized-index function the analytical lane computes.
        from bayesfilter.highdim.ledh_canonical_autodiff_oracle_tf import (
            oracle_forward_autodiff_score,
        )

        theta_np = theta.numpy()

        def value_fn_1p(theta_1):
            parts = [
                theta_1[0] if i == direction_index
                else tf.constant(theta_np[i], DTYPE)
                for i in range(p_count)
            ]
            value, _ = run(tf.stack(parts), seeds[0], False)
            return value

        reference = float(
            oracle_forward_autodiff_score(
                value_fn_1p,
                tf.constant([theta_np[direction_index]], DTYPE),
            )[0].numpy()
        )
        reference_kind = "oracle_seed0"
    else:
        h = 1.0e-5
        up = theta.numpy().copy(); up[direction_index] += h
        down = theta.numpy().copy(); down[direction_index] -= h
        v_up, _ = run(tf.constant(up, DTYPE), seeds[0], False)
        v_down, _ = run(tf.constant(down, DTYPE), seeds[0], False)
        reference = (float(v_up.numpy()) - float(v_down.numpy())) / (2 * h)
        reference_kind = "central_fd_seed0"
    return {
        "program": program,
        "tuning": tuning,
        "crashed_seeds": crashed,
        "all_finite": bool(np.all(np.isfinite(scores))),
        "direction_index": direction_index,
        "analytical_scores": scores,
        "mean": float(np.mean(scores)),
        "seed_spread": float(np.std(scores)),
        "self_consistency_reference": reference,
        "self_consistency_kind": reference_kind,
        "self_consistency_rel_err_seed0": abs(analytical_seed0 - reference)
        / max(abs(reference), 1.0),
        "annealed_stages": annealed_stages,
        "note": "reference is explanatory (same estimator, same seed); "
        "analytical spread is particle-seed variation",
    }


def summarize(values, valids, program="production v1 (registry: contract_e reset + dual-cap "
              "trust region ON)",
              tuning=UNTUNED):
    return {
        "program": program,
        "tuning": tuning,
        "values": values,
        "mean": float(np.mean(values)),
        "seed_spread": float(np.std(values)),
        "all_valid": bool(all(valids)),
        "all_finite": bool(np.all(np.isfinite(values))),
        "seeds": len(values),
    }


# ------------------------------------------------------------------- rows
def row_linear2d_fixture():
    from test_ledh_canonical_filter import (
        _callbacks_for_lgssm,
        _kalman_log_likelihood,
        _lgssm_model,
    )

    spec = _lgssm_model(101)
    exact = _kalman_log_likelihood(spec)
    callbacks = _callbacks_for_lgssm(spec)
    observations = tf.constant(spec["observations"], DTYPE)
    values, valids, _ess = canonical_filter_values(
        callbacks, observations, VALUE_SEEDS
    )
    transition = tf.constant(spec["transition"], DTYPE)

    class _LG:
        transition_mean_fn = staticmethod(
            lambda theta, p: tf.einsum("ij,nj->ni", transition, p)
        )
        observation_fn = staticmethod(lambda p: p)
        observation_jacobian_fn = staticmethod(
            lambda p: tf.broadcast_to(
                tf.constant(spec["obs_matrix"], DTYPE),
                [tf.shape(p)[0], 2, 2],
            )
        )
        process_covariance = tf.constant(spec["process_cov"], DTYPE)
        observation_covariance = tf.constant(spec["obs_cov"], DTYPE)
        observation_log_density_fn = None

    obs_log = _gaussian_obs_log(_LG, None)
    initial_mean = tf.constant(spec["initial_mean"], DTYPE)
    boot = [
        bootstrap_value(_LG, None, observations, obs_log, initial_mean,
                        2, 1008, seed)
        for seed in VALUE_SEEDS
    ]
    ukf = ukf_gaussian_filter_value(
        _LG, None, initial_mean,
        tf.constant(spec["initial_cov"], DTYPE), observations,
    )
    return {
        "data": "test-fixture linear 2d, T=%d" % int(observations.shape[0]),
        "exact_kalman": exact,
        "canonical_ledh": {
            **summarize(values, valids),
            "abs_error_of_mean_vs_exact": abs(float(np.mean(values)) - exact),
        },
        "bootstrap_pf": {
            **summarize(boot, [True] * len(boot), program="comparator (bootstrap PF, systematic resampling)", tuning="N/A (no tunables beyond N)"),
            "abs_error_of_mean_vs_exact": abs(float(np.mean(boot)) - exact),
        },
        "ukf_gaussian_filter": {
            "value": ukf,
            "abs_error_vs_exact": abs(ukf - exact),
            "note": "UKF == Kalman on linear models (sanity anchor)",
        },
    }


def row_diagonal_lgssm():
    from bayesfilter.highdim.ledh_canonical_models_tf import (
        diagonal_lgssm_canonical_model,
    )
    from bayesfilter.highdim.ledh_canonical_neutra_targets_tf import (
        _lgssm_frozen_observations,
    )

    theta0_np = [0.9, 0.8, 0.7, 0.6, 0.8]
    theta0 = tf.constant(theta0_np, DTYPE)
    model, set_direction = diagonal_lgssm_canonical_model(theta0)
    observations = tf.cast(_lgssm_frozen_observations(), DTYPE)
    obs_matrix = np.array(
        [[1.0, 0.25, -0.15], [0.2, 1.1, 0.3], [-0.1, 0.35, 0.9]]
    )

    def exact_kalman(theta_np):
        phi = np.diag(theta_np[:3])
        q = theta_np[3] ** 2 * np.eye(3)
        r = theta_np[4] ** 2 * np.eye(3)
        mean, cov = np.zeros(3), np.eye(3)
        total = 0.0
        for t in range(int(observations.shape[0])):
            mean = phi @ mean
            cov = phi @ cov @ phi.T + q
            s = obs_matrix @ cov @ obs_matrix.T + r
            resid = observations.numpy()[t] - obs_matrix @ mean
            sign, logdet = np.linalg.slogdet(2.0 * np.pi * s)
            total += -0.5 * (resid @ np.linalg.solve(s, resid) + logdet)
            gain = cov @ obs_matrix.T @ np.linalg.inv(s)
            mean = mean + gain @ resid
            cov = (np.eye(3) - gain @ obs_matrix) @ cov
        return float(total)

    exact = exact_kalman(np.array(theta0_np))
    h = 1.0e-6
    up = np.array(theta0_np); up[0] += h
    down = np.array(theta0_np); down[0] -= h
    exact_score_dir0 = (exact_kalman(up) - exact_kalman(down)) / (2 * h)

    initial_mean = tf.zeros([3], DTYPE)
    callbacks = make_callbacks(
        model, theta0, "q3_diagonal_lgssm", 3, 3, initial_mean
    )
    values, valids, _ess = canonical_filter_values(
        callbacks, observations, VALUE_SEEDS
    )
    obs_log = _density_obs_log(model, theta0)
    boot = [
        bootstrap_value(model, theta0, observations, obs_log,
                        initial_mean, 3, 1008, seed)
        for seed in VALUE_SEEDS
    ]
    ukf = ukf_gaussian_filter_value(
        model, theta0, initial_mean, tf.eye(3, dtype=DTYPE), observations
    )
    score = score_cells(
        model, set_direction, theta0, 3, observations, 0, SCORE_SEEDS,
        reset_kwargs=production_score_kwargs(3),
        program="production v1 (registry: S6 contract_e reset + S7 dual-cap trust region)",
    )
    score["exact_kalman_fd_score_dir0"] = exact_score_dir0
    score["abs_error_of_mean_vs_exact"] = abs(
        score["mean"] - exact_score_dir0
    )
    return {
        "data": "frozen benchmark_lgssm_m3_T50_seed81100",
        "exact_kalman": exact,
        "canonical_ledh": {
            **summarize(values, valids),
            "abs_error_of_mean_vs_exact": abs(float(np.mean(values)) - exact),
        },
        "bootstrap_pf": {
            **summarize(boot, [True] * len(boot), program="comparator (bootstrap PF, systematic resampling)", tuning="N/A (no tunables beyond N)"),
            "abs_error_of_mean_vs_exact": abs(float(np.mean(boot)) - exact),
        },
        "ukf_gaussian_filter": {
            "value": ukf,
            "abs_error_vs_exact": abs(ukf - exact),
        },
        "score_dir0": score,
    }


def row_predator_prey():
    from bayesfilter.highdim.ledh_canonical_models_tf import (
        predator_prey_canonical_model,
    )
    from bayesfilter.testing.predator_prey_ukf_neutra_target_tf import (
        generate_frozen_predator_prey_dataset_tf,
    )

    theta0 = tf.constant([0.8, 90.0, 25.0, 0.5, 0.4, 0.3], DTYPE)
    model, set_direction = predator_prey_canonical_model(theta0)
    _states, observations64 = generate_frozen_predator_prey_dataset_tf()
    observations = tf.cast(observations64, DTYPE)
    initial_mean = tf.constant([50.0, 5.0], DTYPE)
    callbacks = make_callbacks(
        model, theta0, "q3_predator_prey", 2,
        int(model.observation_covariance.shape[0]), initial_mean,
    )
    values, valids, _ess = canonical_filter_values(
        callbacks, observations, VALUE_SEEDS
    )
    obs_log = _density_obs_log(model, theta0)
    boot = [
        bootstrap_value(model, theta0, observations, obs_log,
                        initial_mean, 2, 1008, seed)
        for seed in VALUE_SEEDS
    ]
    ukf = ukf_gaussian_filter_value(
        model, theta0, initial_mean, tf.eye(2, dtype=DTYPE), observations
    )
    score = score_cells(
        model, set_direction, theta0, 2, observations, 0, SCORE_SEEDS,
        reset_kwargs=production_score_kwargs(2),
        program="production v1 (registry: S6 contract_e reset + S7 dual-cap trust region)",
    )
    return {
        "data": "frozen predator_prey_T20",
        "canonical_ledh": summarize(values, valids),
        "bootstrap_pf": summarize(boot, [True] * len(boot), program="comparator (bootstrap PF, systematic resampling)", tuning="N/A (no tunables beyond N)"),
        "ukf_gaussian_filter": {
            "value": ukf,
            "note": "Gaussian-approximation comparator",
        },
        "score_dir0": score,
    }


def row_ksc_sv(value_seeds, score_seeds):
    from bayesfilter.highdim.ledh_canonical_models_tf import (
        ksc_sv_canonical_model,
    )
    from bayesfilter.testing.exact_sv_sgqf_neutra_target_tf import (
        generate_frozen_exact_sv_dataset_tf,
    )
    from bayesfilter.testing.ksc_ukf_neutra_target_tf import (
        transformed_ksc_observations,
    )

    theta0 = tf.constant([0.5, 0.1], DTYPE)
    model, set_direction = ksc_sv_canonical_model(theta0)
    _states, raw = generate_frozen_exact_sv_dataset_tf()
    observations = tf.cast(transformed_ksc_observations(raw), DTYPE)
    if observations.shape.rank == 1:
        observations = observations[:, None]
    initial_mean = tf.zeros([1], DTYPE)
    callbacks = make_callbacks(
        model, theta0, "q3_ksc_sv", 1, 1, initial_mean
    )
    values, valids, _ess = canonical_filter_values(
        callbacks, observations, value_seeds
    )
    obs_log = _density_obs_log(model, theta0)
    boot = [
        bootstrap_value(model, theta0, observations, obs_log,
                        initial_mean, 1, 1008, seed)
        for seed in value_seeds
    ]
    ukf = ukf_gaussian_filter_value(
        model, theta0, initial_mean, tf.eye(1, dtype=DTYPE), observations
    )
    score = score_cells(
        model, set_direction, theta0, 1, observations, 0, score_seeds,
        reset_kwargs=production_score_kwargs(1),
        program="production v1 (registry: S6 contract_e reset + S7 dual-cap trust region)",
    )
    return {
        "data": "frozen zhao_cui_sv_ksc_T1000 (full mixture horizon)",
        "canonical_ledh": summarize(values, valids),
        "bootstrap_pf": summarize(boot, [True] * len(boot), program="comparator (bootstrap PF, systematic resampling)", tuning="N/A (no tunables beyond N)"),
        "ukf_gaussian_filter": {
            "value": ukf,
            "note": "Gaussian-approximation comparator; density-"
            "misspecified vs the KSC mixture (recorded)",
        },
        "score_dir0": score,
    }


def row_generalized_sv():
    from bayesfilter.highdim.ledh_canonical_models_tf import (
        generalized_sv_canonical_model,
    )

    theta0_np = [
        float(np.arctanh(0.7)),
        float(np.arctanh(0.6)),
        -0.4,
        -0.6,
        0.1,
    ]
    theta0 = tf.constant(theta0_np, DTYPE)
    model, set_direction = generalized_sv_canonical_model(theta0)
    # simulate from the gen-SV law (fisher-harness convention)
    rng = np.random.default_rng(501)
    rho_s, rho_h = np.tanh(theta0_np[0]), np.tanh(theta0_np[1])
    sigma_s, sigma_h = np.exp(theta0_np[2]), np.exp(theta0_np[3])
    beta = np.exp(theta0_np[4])
    s = rng.normal(0.0, sigma_s / np.sqrt(1.0 - rho_s**2))
    hh = rng.normal(0.0, sigma_h / np.sqrt(1.0 - rho_h**2))
    obs_list = []
    for _t in range(20):
        s = rho_s * s + sigma_s * rng.standard_normal()
        hh = rho_h * hh + sigma_h * rng.standard_normal()
        obs_list.append(beta * s + np.exp(0.5 * hh) * rng.standard_normal())
    observations = tf.constant(np.array(obs_list)[:, None], DTYPE)
    initial_mean = tf.zeros([2], DTYPE)
    callbacks = make_callbacks(
        model, theta0, "q3_generalized_sv", 2, 1, initial_mean
    )
    values, valids, _ess = canonical_filter_values(
        callbacks, observations, VALUE_SEEDS
    )
    obs_log = _density_obs_log(model, theta0)
    boot = [
        bootstrap_value(model, theta0, observations, obs_log,
                        initial_mean, 2, 1008, seed)
        for seed in VALUE_SEEDS
    ]
    ukf = ukf_gaussian_filter_value(
        model, theta0, initial_mean, tf.eye(2, dtype=DTYPE), observations
    )
    score = score_cells(
        model, set_direction, theta0, 2, observations, 4, SCORE_SEEDS,
        reset_kwargs=production_score_kwargs(2),
        program="production v1 (registry: S6 contract_e reset + S7 dual-cap trust region)",
    )
    return {
        "data": "simulated gen-SV T=20 (seed 501; heteroskedastic law)",
        "canonical_ledh": summarize(values, valids),
        "bootstrap_pf": summarize(boot, [True] * len(boot), program="comparator (bootstrap PF, systematic resampling)", tuning="N/A (no tunables beyond N)"),
        "ukf_gaussian_filter": {
            "value": ukf,
            "note": "Gaussian-approximation comparator; density-"
            "misspecified vs heteroskedastic law (recorded)",
        },
        "score_dir4": score,
    }


def row_austria_sir():
    from bayesfilter.highdim.ledh_canonical_models_tf import (
        austria_sir_canonical_model,
    )
    from bayesfilter.highdim.ledh_canonical_neutra_targets_tf import (
        make_canonical_neutra_target,
    )
    from bayesfilter.highdim.models import zhao_cui_sir_austria_model

    with tf.device("/CPU:0"):
        target = make_canonical_neutra_target(
            "austria_sir", particle_count=1008
        )
    theta0 = tf.constant([0.0, 0.0, 0.0], DTYPE)
    model, set_direction = austria_sir_canonical_model(theta0)
    observations = tf.cast(target.observations, DTYPE)
    initial_mean = tf.cast(zhao_cui_sir_austria_model().initial_mean, DTYPE)
    callbacks = make_callbacks(
        model, theta0, "q3_austria_sir", 18, 9, initial_mean
    )
    # Q2-calibrated annealed arm (f64 anchor semantics on this row);
    # the f32/TF32 GPU lane result is the Curve-1 artifact (linked in
    # the report) — value cells here replicate the f64 anchor at 16
    # seeds for the leaderboard's seed-spread column.
    values, valids, min_ess = canonical_filter_values(
        callbacks, observations, VALUE_SEEDS,
        temper_stages=4, annealed_resampling=True, flow_prior_cap=8.0,
    )
    obs_log = _density_obs_log(model, theta0)
    boot = [
        bootstrap_value(model, theta0, observations, obs_log,
                        initial_mean, 18, 1008, seed)
        for seed in VALUE_SEEDS
    ]
    ukf = ukf_gaussian_filter_value(
        model, theta0, initial_mean, tf.eye(18, dtype=DTYPE), observations
    )
    score = score_cells(
        model, set_direction, theta0, 18, observations, 0, SCORE_SEEDS,
        annealed_stages=4, initial_mean=initial_mean,
        self_consistency="oracle",
        reset_kwargs=production_score_kwargs(18),
        program="production v1 (registry + annealed k=4 per Q2 calibration)",
        tuning="Q2 Curve-1 artifact (k=4/c=8) + Curve-3 damping; "
        "epsilon/substeps/reset controls untuned",
    )
    return {
        "data": "frozen austria_sir_y1_y20; Q2-calibrated annealed "
        "k=4/c=8 canonical arm (per-scope calibration applies to this "
        "row only)",
        "canonical_ledh": {
            **summarize(values, valids),
            "min_stage_ess_fraction": float(np.min(min_ess)) / 1008.0,
        },
        "bootstrap_pf": summarize(boot, [True] * len(boot), program="comparator (bootstrap PF, systematic resampling)", tuning="N/A (no tunables beyond N)"),
        "ukf_gaussian_filter": {"value": ukf},
        "score_dir0": score,
    }


ROWS = {
    "linear2d": row_linear2d_fixture,
    "dlgssm": row_diagonal_lgssm,
    "predator_prey": row_predator_prey,
    "ksc_sv": None,  # dispatched with scale args
    "generalized_sv": row_generalized_sv,
    "austria_sir": row_austria_sir,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row", required=True, choices=list(ROWS))
    parser.add_argument("--skip-conformance", action="store_true",
                        help="debug only; artifact is NOT stamped")
    parser.add_argument("--ksc-value-seeds", type=int, default=16)
    parser.add_argument("--ksc-score-seeds", type=int, default=8)
    args = parser.parse_args()
    started = time.time()

    gpus = tf.config.list_physical_devices("GPU")

    from bayesfilter.highdim.ledh_alg1_contract import (
        ALG1_CONFORMANCE_SUITE_VERSION,
    )

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=_ROOT, capture_output=True,
        text=True,
    ).stdout.strip()

    stamp = None
    if not args.skip_conformance:
        battery = subprocess.run(
            [
                sys.executable, "-m", "pytest", "-q", "-x",
                "tests/highdim/test_ledh_canonical_governance.py",
                "tests/highdim/test_ledh_canonical_meta_governance.py",
                "tests/highdim/test_ledh_canonical_score_full.py",
            ],
            cwd=_ROOT, capture_output=True, text=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
        )
        if battery.returncode != 0:
            print(battery.stdout[-2000:])
            raise SystemExit(
                "FAIL-CLOSED: conformance battery not green; no stamp, "
                "no row"
            )
        stamp = f"{ALG1_CONFORMANCE_SUITE_VERSION}@{commit[:12]}"

    if args.row == "ksc_sv":
        row = row_ksc_sv(
            list(range(args.ksc_value_seeds)),
            list(range(args.ksc_score_seeds)),
        )
    else:
        row = ROWS[args.row]()

    payload = {
        "schema": "bayesfilter.q3_leaderboard_row.v1",
        "plan": "docs/plans/bayesfilter-q3-leaderboard-execution-plan-2026-08-24.md",
        "row": args.row,
        "alg1_conformance": stamp,
        "manifest": {
            "commit": commit,
            "command": " ".join(sys.argv),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
            "cuda_visible_devices": os.environ.get(
                "CUDA_VISIBLE_DEVICES", "unset"
            ),
            "gpus_visible": [g.name for g in gpus],
            "dtype": "float64",
            "particle_count": 1008,
            "value_seeds": len(VALUE_SEEDS),
            "score_seeds": len(SCORE_SEEDS),
            "wall_seconds": round(time.time() - started, 1),
        },
        "result": row,
    }
    out_dir = os.path.join(
        _ROOT, "docs", "benchmarks", "artifacts",
        "ledh_canonical_leaderboard_2026-08", "q3_board",
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"row_{args.row}.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
    print(
        f"[done] row={args.row} stamped={stamp is not None} "
        f"wall={time.time() - started:.0f}s artifact={out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
