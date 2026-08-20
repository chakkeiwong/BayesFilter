"""P4 full-recursion gate: multi-step analytical score vs oracle (LGSSM).

The complete T-step canonical value program with the analytical score
recursion, including tangent propagation through the state recursion and
the OT reset (S6: the reset consumes weighted children; its tangent uses
the hand-derived reset JVP pattern — here gated on the simplified
equal-weight reset path the canonical filter uses after each step).

The multi-step program uses UNIFORM reset-to-children semantics for the
tangent recursion slice being gated (reset_policy='none'): children carry
to the next step directly. This isolates S1-S5+S8 correctness over the
full recursion; the Contract-E reset tangent (S6) is gated separately
before entering the assembled score module.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_autodiff_oracle_tf import (
    oracle_forward_autodiff_score,
)
from bayesfilter.highdim.ledh_canonical_score_scaffold_tf import (
    multi_step_value_and_score_lgssm,
)

DTYPE = tf.float64


def test_multi_step_analytical_score_matches_oracle():
    rng = np.random.default_rng(41)
    n, dim, horizon = 8, 2, 4
    f0 = tf.constant(np.array([[0.9, 0.1], [0.05, 0.8]]), DTYPE)
    q = tf.constant(0.4 * np.eye(dim), DTYPE)
    h = tf.constant(np.eye(dim), DTYPE)
    r = tf.constant(0.6 * np.eye(dim), DTYPE)
    initial = tf.constant(rng.standard_normal((n, dim)), DTYPE)
    covs = tf.constant(np.stack([np.eye(dim)] * n), DTYPE)
    noises = tf.constant(rng.standard_normal((horizon, n, dim)), DTYPE)
    observations = tf.constant(rng.standard_normal((horizon, dim)), DTYPE)
    theta0 = tf.constant([1.0], DTYPE)

    def value_fn(theta):
        value, _ = multi_step_value_and_score_lgssm(
            theta, f0, q, h, r, initial, covs, noises, observations,
            substeps=12, with_tangent=False,
        )
        return value

    oracle = oracle_forward_autodiff_score(value_fn, theta0)
    _, score = multi_step_value_and_score_lgssm(
        theta0, f0, q, h, r, initial, covs, noises, observations,
        substeps=12, with_tangent=True,
    )
    err = abs(float(score[0].numpy()) - float(oracle[0].numpy()))
    scale = max(abs(float(oracle[0].numpy())), 1.0)
    assert err < 1.0e-4 * scale, (
        f"multi-step analytical {float(score[0].numpy())} vs oracle "
        f"{float(oracle[0].numpy())}"
    )
