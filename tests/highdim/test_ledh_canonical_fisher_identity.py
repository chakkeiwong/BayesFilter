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
