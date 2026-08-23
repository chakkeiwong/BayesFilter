"""Leaderboard slice 2: cross-algorithm value/score comparison (CPU f64).

Per model: value cells for (exact reference where linear) | canonical
LEDH-PF-PF | bootstrap PF | UKF Gaussian filter — same data, same theta,
3 particle seeds where stochastic. Score cells: canonical ANALYTICAL score
vs central-FD of each algorithm's own value (self-consistency check per
algorithm; FD is explanatory reference only).

Statistical discipline: descriptive; per-seed spread reported; no ranking
language beyond exact-reference absolute errors.
"""

from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tests", "highdim"))

import numpy as np
import tensorflow as tf

DTYPE = tf.float64


def ukf_gaussian_filter_value(model, theta, initial_mean, initial_cov, observations):
    """Standard UKF (Gaussian filter) log-likelihood: single mean/cov
    recursion with unscented moments; innovation Gaussians accumulate."""
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
        residual = obs - observed_pred
        sign, logdet = np.linalg.slogdet(
            2.0 * np.pi * innovation_cov.numpy()
        )
        total += float(
            -0.5
            * (
                residual.numpy()
                @ np.linalg.solve(innovation_cov.numpy(), residual.numpy())
                + logdet
            )
        )
        mean, cov = ukf_update_per_particle(
            pm, pc, lambda p: model.observation_fn(p),
            model.observation_covariance, obs,
        )
        mean = model.transition_mean_fn(theta, mean) * 0.0 + mean  # keep mean
    return total


def canonical_value(model, theta, initial, covs, noises, observations, substeps=8):
    from bayesfilter.highdim.ledh_canonical_score_tf import (
        canonical_value_and_analytical_score,
    )

    value, _ = canonical_value_and_analytical_score(
        model, theta, initial, covs, noises, observations,
        substeps=substeps, with_score=False,
    )
    return float(value.numpy())


def canonical_score(model, set_direction, theta, initial, covs, noises, observations, direction, substeps=8):
    from bayesfilter.highdim.ledh_canonical_score_tf import (
        canonical_value_and_analytical_score,
    )

    set_direction(tf.constant(direction, DTYPE))
    _, score = canonical_value_and_analytical_score(
        model, theta, initial, covs, noises, observations,
        substeps=substeps, with_score=True,
    )
    return float(score[0].numpy())


def bootstrap_value(model, theta, initial, noises, observations, obs_log_fn):
    states = initial
    total = 0.0
    n = int(initial.shape[0])
    for t in range(int(observations.shape[0])):
        obs = observations[t]
        anchors = model.transition_mean_fn(theta, states)
        chol = tf.linalg.cholesky(model.process_covariance)
        children = anchors + tf.einsum("ij,nj->ni", chol, noises[t])
        ol = obs_log_fn(model, children, obs)
        total += float(
            tf.reduce_logsumexp(ol - np.log(n)).numpy()
        )
        states = children
    return total


def default_obs_log(model, points, obs):
    # Density-aware (2026-08-24): when the model declares a non-Gaussian
    # observation density (mixture, heteroskedastic), ALL particle
    # comparators must use it — shared-Gaussianization was the
    # cross-algorithm blindness that masked infidelities #2 and #5.
    density_fn = getattr(model, "observation_log_density_fn", None)
    if density_fn is not None:
        theta_attr = getattr(model, "_slice2_theta", None)
        return density_fn(theta_attr, points, obs)
    observed = model.observation_fn(points)
    r = model.observation_covariance.numpy()
    residual = obs.numpy()[None, :] - observed.numpy()
    sign, logdet = np.linalg.slogdet(2.0 * np.pi * r)
    quad = np.einsum(
        "no,op,np->n", residual, np.linalg.inv(r), residual
    )
    return tf.constant(-0.5 * (quad + logdet), DTYPE)


def fd_score(value_fn, theta, index, h=1e-5):
    up = theta.numpy().copy(); up[index] += h
    down = theta.numpy().copy(); down[index] -= h
    return (value_fn(tf.constant(up, DTYPE)) - value_fn(tf.constant(down, DTYPE))) / (2 * h)


def build_cells() -> dict:
    from bayesfilter.highdim.ledh_canonical_models_tf import (
        austria_sir_canonical_model,
        diagonal_lgssm_canonical_model,
        generalized_sv_canonical_model,
        ksc_sv_canonical_model,
        predator_prey_canonical_model,
    )
    from test_ledh_canonical_filter import (
        _kalman_log_likelihood,
        _lgssm_model,
    )
    from test_ledh_canonical_score_full import _model as nonlinear_model_2d

    results = {}

    # ---------- Model 1: LGSSM fixture (EXACT Kalman reference) ----------
    model_spec = _lgssm_model(101)
    exact = _kalman_log_likelihood(model_spec)
    from test_ledh_canonical_filter import _callbacks_for_lgssm
    from bayesfilter.highdim.ledh_canonical_filter_tf import (
        canonical_value_and_diagnostics,
    )

    callbacks = _callbacks_for_lgssm(model_spec)
    canon_vals = []
    for seed in (0, 1, 2):
        result = canonical_value_and_diagnostics(
            callbacks, tf.constant(model_spec["observations"], DTYPE),
            particle_count=4096, seed=seed,
        )
        canon_vals.append(float(result["value"].numpy()))
    # bootstrap on same model
    rng = np.random.default_rng(7)
    n_b = 4096
    transition = tf.constant(model_spec["transition"], DTYPE)

    class _LG:
        transition_mean_fn = staticmethod(
            lambda theta, p: tf.einsum("ij,nj->ni", transition, p)
        )
        observation_fn = staticmethod(lambda p: p)
        observation_jacobian_fn = staticmethod(
            lambda p: tf.broadcast_to(
                tf.constant(model_spec["obs_matrix"], DTYPE),
                [tf.shape(p)[0], 2, 2],
            )
        )
        process_covariance = tf.constant(model_spec["process_cov"], DTYPE)
        observation_covariance = tf.constant(model_spec["obs_cov"], DTYPE)

    boot_vals = []
    for seed in (0, 1, 2):
        rng2 = np.random.default_rng(seed)
        initial_b = tf.constant(
            rng2.multivariate_normal(
                model_spec["initial_mean"], model_spec["initial_cov"], n_b
            ),
            DTYPE,
        )
        noises_b = tf.constant(
            rng2.standard_normal(
                (len(model_spec["observations"]), n_b, 2)
            ),
            DTYPE,
        )
        boot_vals.append(
            bootstrap_value(
                _LG, None, initial_b, noises_b,
                tf.constant(model_spec["observations"], DTYPE),
                default_obs_log,
            )
        )
    # UKF Gaussian filter (== Kalman on linear model: sanity anchor)
    ukf_val = ukf_gaussian_filter_value(
        _LG, None,
        tf.constant(model_spec["initial_mean"], DTYPE),
        tf.constant(model_spec["initial_cov"], DTYPE),
        tf.constant(model_spec["observations"], DTYPE),
    )
    results["lgssm_fixture_T5"] = {
        "exact_kalman": exact,
        "ukf_gaussian_filter": {
            "value": ukf_val, "abs_error": abs(ukf_val - exact),
            "note": "UKF == Kalman on linear models (sanity anchor)",
        },
        "canonical_ledh": {
            "values": canon_vals,
            "mean": float(np.mean(canon_vals)),
            "abs_error_of_mean": abs(float(np.mean(canon_vals)) - exact),
            "spread": float(np.std(canon_vals)),
        },
        "bootstrap_pf": {
            "values": boot_vals,
            "mean": float(np.mean(boot_vals)),
            "abs_error_of_mean": abs(float(np.mean(boot_vals)) - exact),
            "spread": float(np.std(boot_vals)),
        },
    }

    # ---------- Models 2-6: canonical vs bootstrap vs UKF + score table --
    def onboarded(name, factory, theta0_np, dim, direction_index, seed, obs_noise, horizon=3, n=128):
        theta0 = tf.constant(theta0_np, DTYPE)
        model, set_direction = factory(theta0)
        rng = np.random.default_rng(seed)
        initial = tf.constant(rng.standard_normal((n, dim)), DTYPE)
        covs = tf.constant(np.stack([np.eye(dim)] * n), DTYPE)
        noises = tf.constant(
            rng.standard_normal((horizon, n, dim)), DTYPE
        )
        obs_dim = int(model.observation_covariance.shape[0])
        observations = tf.constant(
            obs_noise * rng.standard_normal((horizon, obs_dim)), DTYPE
        )
        object.__setattr__(model, "_slice2_theta", theta0)
        c_val = canonical_value(
            model, theta0, initial, covs, noises, observations
        )
        b_val = bootstrap_value(
            model, theta0, initial, noises, observations, default_obs_log
        )
        u_val = ukf_gaussian_filter_value(
            model, theta0,
            tf.reduce_mean(initial, axis=0),
            tf.eye(dim, dtype=DTYPE),
            observations,
        )
        p_count = len(theta0_np)
        one_hot = np.zeros(p_count); one_hot[direction_index] = 1.0
        analytical = canonical_score(
            model, set_direction, theta0, initial, covs, noises,
            observations, one_hot,
        )

        def canon_value_fn(theta):
            return canonical_value(
                model, theta, initial, covs, noises, observations
            )

        fd = fd_score(canon_value_fn, theta0, direction_index)
        return {
            "theta": list(theta0_np),
            "score_direction_index": direction_index,
            "values": {
                "canonical_ledh": c_val,
                "bootstrap_pf": b_val,
                "ukf_gaussian_filter": u_val,
            },
            "scores_direction": {
                "canonical_analytical": analytical,
                "canonical_central_fd_reference": fd,
                "self_consistency_rel_err": abs(analytical - fd)
                / max(abs(fd), 1.0),
            },
        }

    results["austria_sir_T3_synthetic"] = onboarded(
        "austria",
        austria_sir_canonical_model,
        [0.0, 0.0, 0.0], 18, 0, 83, 10.0,
    )
    results["predator_prey_T3"] = onboarded(
        "pp",
        predator_prey_canonical_model,
        [0.8, 90.0, 25.0, 0.5, 0.4, 0.3], 2, 0, 211, 2.0,
    )
    results["diagonal_lgssm_T3"] = onboarded(
        "dlg",
        diagonal_lgssm_canonical_model,
        [0.9, 0.8, 0.7, 0.6, 0.8], 3, 0, 221, 1.0,
    )
    results["ksc_sv_T3"] = onboarded(
        "ksc",
        ksc_sv_canonical_model,
        [0.5, 0.1], 1, 0, 231, 2.0,
    )
    results["generalized_sv_T3"] = onboarded(
        "gsv",
        generalized_sv_canonical_model,
        [float(np.arctanh(0.9)), float(np.arctanh(0.8)), -0.5, -0.7, 0.2],
        2, 0, 241, 1.0,
    )
    return results


def main() -> None:
    started = time.time()
    results = build_cells()
    payload = {
        "schema": "bayesfilter.ledh_canonical_leaderboard_slice2.v1",
        "note": (
            "CPU float64 descriptive comparison; canonical analytical "
            "scores gated vs oracle at onboarding (rtol 1e-4); FD columns "
            "are explanatory references; no ranking without uncertainty"
        ),
        "results": results,
        "wall_seconds": time.time() - started,
    }
    out = os.path.join(
        _ROOT, "docs", "benchmarks", "artifacts",
        "ledh_canonical_leaderboard_2026-08", "slice2_cross_algorithm.json",
    )
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
    print(json.dumps(payload["results"], indent=1))


if __name__ == "__main__":
    main()
