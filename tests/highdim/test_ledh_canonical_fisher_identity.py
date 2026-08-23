"""C-class statistical-identity gate: Fisher identity (template: gen-SV).

E_{data ~ p(.|theta*)}[ score(theta*) ] = 0 for the exact likelihood
score. The particle estimate adds O(1/N) bias and MC spread, so the gate
is statistical with declared tolerance: |mean score| < 3*SE + bias slack.

WHY THIS GATE CLASS (2026-08-24): it tests observation-density fidelity
and score correctness JOINTLY against only the model's own SIMULATOR — an
independent, simpler code path. The gen-SV heteroskedastic-density defect
(fixed 2026-08-23) is exactly the class this catches without needing any
independent density implementation: simulating y ~ N(beta*s, exp(h)) and
scoring with a wrong-family density (e.g. fixed variance 1) leaves a
nonzero mean score in the variance-bearing directions.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_models_tf import (
    generalized_sv_canonical_model,
)
from bayesfilter.highdim.ledh_canonical_score_tf import (
    canonical_value_and_analytical_score,
)

DTYPE = tf.float64


def _simulate_dataset(rng, theta_np, horizon):
    """Simulate from the reference generative law (independent simple
    path: raw numpy, no shared code with the density under test)."""
    rho_s = np.tanh(theta_np[0])
    rho_h = np.tanh(theta_np[1])
    sigma_s = np.exp(theta_np[2])
    sigma_h = np.exp(theta_np[3])
    beta = np.exp(theta_np[4])
    s = rng.normal(0.0, sigma_s / np.sqrt(1.0 - rho_s**2))
    h = rng.normal(0.0, sigma_h / np.sqrt(1.0 - rho_h**2))
    observations = []
    states = []
    for _ in range(horizon):
        s = rho_s * s + sigma_s * rng.standard_normal()
        h = rho_h * h + sigma_h * rng.standard_normal()
        observations.append(
            beta * s + np.exp(0.5 * h) * rng.standard_normal()
        )
        states.append((s, h))
    return np.array(observations)[:, None], states


def test_generalized_sv_fisher_identity():
    theta_np = [
        float(np.arctanh(0.7)),
        float(np.arctanh(0.6)),
        -0.4,
        -0.6,
        0.1,
    ]
    theta = tf.constant(theta_np, DTYPE)
    model, set_direction = generalized_sv_canonical_model(theta)
    horizon, n, replications = 3, 192, 40
    rng = np.random.default_rng(401)

    # direction: theta_4 (log beta) — the direction the historical
    # wrong-density defect corrupts most directly (obs mean AND variance
    # family), plus theta_3 (log sigma_h) which is variance-bearing.
    for direction_index in (4, 3):
        one_hot = np.zeros(5)
        one_hot[direction_index] = 1.0
        set_direction(tf.constant(one_hot, DTYPE))
        scores = []
        for rep in range(replications):
            observations_np, _ = _simulate_dataset(rng, theta_np, horizon)
            # fresh particle randomness per replication
            rng_p = np.random.default_rng(1000 + rep)
            sigma_s = np.exp(theta_np[2])
            sigma_h = np.exp(theta_np[3])
            rho_s = np.tanh(theta_np[0])
            rho_h = np.tanh(theta_np[1])
            init_scale = np.array(
                [
                    sigma_s / np.sqrt(1.0 - rho_s**2),
                    sigma_h / np.sqrt(1.0 - rho_h**2),
                ]
            )
            initial = tf.constant(
                rng_p.standard_normal((n, 2)) * init_scale[None, :], DTYPE
            )
            covs = tf.constant(
                np.stack([np.diag(init_scale**2)] * n), DTYPE
            )
            noises = tf.constant(
                rng_p.standard_normal((horizon, n, 2)), DTYPE
            )
            _, score = canonical_value_and_analytical_score(
                model,
                theta,
                initial,
                covs,
                noises,
                tf.constant(observations_np, DTYPE),
                substeps=6,
                with_score=True,
            )
            scores.append(float(score[0].numpy()))
        scores = np.array(scores)
        mean = float(np.mean(scores))
        se = float(np.std(scores, ddof=1) / np.sqrt(replications))
        # Declared gate: |mean| < 3*SE + 0.05 bias slack (O(1/N) particle
        # bias at N=192, T=3). A wrong-family density fails this by O(1).
        assert abs(mean) < 3.0 * se + 0.05, (
            f"Fisher identity violated in direction {direction_index}: "
            f"mean score {mean:.4f}, SE {se:.4f} over {replications} "
            f"replications — density family or score derivation suspect"
        )


def _fisher_gate(model, set_direction, theta, simulate_fn, initial_sampler,
                 directions, n=192, horizon=2, replications=40,
                 bias_slack=0.05, seed0=2000):
    from bayesfilter.highdim.ledh_canonical_score_tf import (
        canonical_value_and_analytical_score,
    )

    p_count = int(theta.shape[0])
    dim = None
    for direction_index in directions:
        one_hot = np.zeros(p_count)
        one_hot[direction_index] = 1.0
        set_direction(tf.constant(one_hot, DTYPE))
        scores = []
        for rep in range(replications):
            rng = np.random.default_rng(seed0 + rep)
            observations_np = simulate_fn(rng, horizon)
            rng_p = np.random.default_rng(seed0 + 10000 + rep)
            initial_np, cov_np = initial_sampler(rng_p, n)
            dim = initial_np.shape[1]
            initial = tf.constant(initial_np, DTYPE)
            covs = tf.constant(np.stack([cov_np] * n), DTYPE)
            noises = tf.constant(
                rng_p.standard_normal((horizon, n, dim)), DTYPE
            )
            _, score = canonical_value_and_analytical_score(
                model, theta, initial, covs, noises,
                tf.constant(observations_np, DTYPE),
                substeps=6, with_score=True,
            )
            scores.append(float(score[0].numpy()))
        scores = np.array(scores)
        mean = float(np.mean(scores))
        se = float(np.std(scores, ddof=1) / np.sqrt(replications))
        assert abs(mean) < 3.0 * se + bias_slack, (
            f"Fisher identity violated in direction {direction_index}: "
            f"mean {mean:.4f}, SE {se:.4f}"
        )


def test_ksc_sv_fisher_identity():
    from bayesfilter.highdim.ledh_canonical_models_tf import (
        ksc_sv_canonical_model,
    )
    from scipy.special import ndtr

    theta_np = [0.5, 0.1]
    theta = tf.constant(theta_np, DTYPE)
    model, set_direction = ksc_sv_canonical_model(theta)
    gamma = float(ndtr(theta_np[0]))
    weights = np.array(
        [0.00730, 0.10556, 0.00002, 0.04395, 0.34001, 0.24566, 0.25750]
    )
    means = np.array(
        [-10.12999, -3.97281, -8.56686, 2.77786, 0.61942, 1.79518, -1.08819]
    ) - 1.2704
    variances = np.array(
        [5.79596, 2.61369, 5.17950, 0.16735, 0.64009, 0.34023, 1.26261]
    )

    def simulate(rng, horizon):
        h = rng.normal(0.0, 1.0 / np.sqrt(1.0 - gamma**2))
        ys = []
        for _ in range(horizon):
            h = gamma * h + rng.standard_normal()
            k = rng.choice(7, p=weights / weights.sum())
            ys.append(
                h + 2.0 * theta_np[1]
                + rng.normal(means[k], np.sqrt(variances[k]))
            )
        return np.array(ys)[:, None]

    def initial_sampler(rng, n):
        scale = 1.0 / np.sqrt(1.0 - gamma**2)
        return rng.normal(0.0, scale, (n, 1)), np.array([[scale**2]])

    _fisher_gate(
        model, set_direction, theta, simulate, initial_sampler,
        directions=(0, 1), seed0=3000,
    )


def test_diagonal_lgssm_fisher_identity():
    from bayesfilter.highdim.ledh_canonical_models_tf import (
        diagonal_lgssm_canonical_model,
    )

    theta_np = [0.9, 0.8, 0.7, 0.6, 0.8]
    theta = tf.constant(theta_np, DTYPE)
    model, set_direction = diagonal_lgssm_canonical_model(theta)
    phi = np.array(theta_np[:3])
    q_scale, r_scale = theta_np[3], theta_np[4]
    matrix = np.array(
        [[1.0, 0.25, -0.15], [0.2, 1.1, 0.3], [-0.1, 0.35, 0.9]]
    )

    def simulate(rng, horizon):
        x = rng.normal(0.0, q_scale / np.sqrt(1.0 - phi**2))
        ys = []
        for _ in range(horizon):
            x = phi * x + q_scale * rng.standard_normal(3)
            ys.append(matrix @ x + r_scale * rng.standard_normal(3))
        return np.array(ys)

    def initial_sampler(rng, n):
        scale = q_scale / np.sqrt(1.0 - phi**2)
        return (
            rng.standard_normal((n, 3)) * scale[None, :],
            np.diag(scale**2),
        )

    _fisher_gate(
        model, set_direction, theta, simulate, initial_sampler,
        directions=(3, 4), seed0=4000,
    )


def test_predator_prey_fisher_identity():
    from bayesfilter.highdim.ledh_canonical_models_tf import (
        predator_prey_canonical_model,
    )

    theta_np = [0.8, 90.0, 25.0, 0.5, 0.4, 0.3]
    theta = tf.constant(theta_np, DTYPE)
    model, set_direction = predator_prey_canonical_model(theta)

    def rk4_np(state):
        r, cap, half, s_r, u_r, v_r = theta_np
        def rhs(x):
            prey, predator = x
            inter = prey * predator / (half + prey)
            return np.array([
                r * prey * (1.0 - prey / cap) - s_r * inter,
                u_r * inter - v_r * predator,
            ])
        x = state
        for _ in range(20):
            k1 = rhs(x)
            k2 = rhs(x + 0.05 * k1)
            k3 = rhs(x + 0.05 * k2)
            k4 = rhs(x + 0.1 * k3)
            x = x + (0.1 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return x

    def simulate(rng, horizon):
        x = np.array([50.0, 5.0]) + rng.standard_normal(2)
        ys = []
        for _ in range(horizon):
            x = rk4_np(x) + 2.0 * rng.standard_normal(2)
            ys.append(x + 2.0 * rng.standard_normal(2))
        return np.array(ys)

    def initial_sampler(rng, n):
        return (
            np.array([50.0, 5.0])[None, :] + rng.standard_normal((n, 2)),
            np.eye(2),
        )

    _fisher_gate(
        model, set_direction, theta, simulate, initial_sampler,
        directions=(0, 3), n=128, replications=32, seed0=5000,
        bias_slack=0.08,
    )
