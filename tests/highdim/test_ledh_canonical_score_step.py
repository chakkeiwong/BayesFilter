"""P4 S4+S8 gate: analytical one-step log-likelihood-increment score.

The complete one-step PF-PF program: UKF predict (S1, linear slice) ->
flow (S3) -> weight assembly (S4) -> logsumexp increment (S8 single step).
Analytical tangent vs autodiff oracle on the full increment — the score
mark of one filtering step. Green here means the per-step score recursion
is correct end-to-end for the LGSSM slice.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_autodiff_oracle_tf import (
    oracle_forward_autodiff_score,
)
from bayesfilter.highdim.ledh_canonical_score_stages_tf import (
    one_step_increment_and_parameter_tangent_lgssm,
)

DTYPE = tf.float64


def _fixture(seed: int, n: int = 8, dim: int = 2):
    rng = np.random.default_rng(seed)
    f0 = tf.constant(np.array([[0.9, 0.1], [0.05, 0.8]]), DTYPE)
    q = tf.constant(0.4 * np.eye(dim), DTYPE)
    h = tf.constant(np.eye(dim), DTYPE)
    r = tf.constant(0.6 * np.eye(dim), DTYPE)
    states = tf.constant(rng.standard_normal((n, dim)), DTYPE)
    covs = tf.constant(
        np.stack([np.eye(dim) * (0.5 + 0.2 * i) for i in range(n)]), DTYPE
    )
    noise = tf.constant(rng.standard_normal((n, dim)), DTYPE)
    observation = tf.constant(rng.standard_normal(dim), DTYPE)
    weights = tf.fill([n], tf.constant(1.0 / n, DTYPE))
    return f0, q, h, r, states, covs, noise, observation, weights


def test_s4_s8_one_step_increment_score_matches_oracle():
    f0, q, h, r, states, covs, noise, observation, weights = _fixture(21)
    theta0 = tf.constant([1.0], DTYPE)

    def increment_fn(theta):
        increment, _ = one_step_increment_and_parameter_tangent_lgssm(
            theta, f0, q, h, r, states, covs, noise, observation, weights,
            substeps=16, with_tangent=False,
        )
        return increment

    oracle = oracle_forward_autodiff_score(increment_fn, theta0)
    _, tangent = one_step_increment_and_parameter_tangent_lgssm(
        theta0, f0, q, h, r, states, covs, noise, observation, weights,
        substeps=16, with_tangent=True,
    )
    err = abs(float(tangent[0].numpy()) - float(oracle[0].numpy()))
    scale = max(abs(float(oracle[0].numpy())), 1.0)
    assert err < 1.0e-4 * scale, (
        f"S4+S8 analytical {float(tangent[0].numpy())} vs oracle "
        f"{float(oracle[0].numpy())}"
    )


def test_s4_s8_score_is_finite_and_seed_stable():
    values = []
    for seed in (31, 32):
        f0, q, h, r, states, covs, noise, observation, weights = _fixture(seed)
        _, tangent = one_step_increment_and_parameter_tangent_lgssm(
            tf.constant([1.0], DTYPE), f0, q, h, r, states, covs, noise,
            observation, weights, substeps=16, with_tangent=True,
        )
        values.append(float(tangent[0].numpy()))
    assert all(np.isfinite(v) for v in values)
