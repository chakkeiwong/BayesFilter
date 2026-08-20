"""P4 S3 gate: analytical flow-map parameter derivative vs autodiff oracle.

LGSSM slice: theta enters through the transition matrix F(theta) = theta*F0.
The flow's A/b depend on theta via P^i(theta) (UKF prediction) and the
anchor. Stage derivation under test: d(post_flow)/dtheta assembled from
the substep chain rule (derivation note S3).
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
    flow_value_and_parameter_tangent_lgssm,
)

DTYPE = tf.float64


def test_s3_flow_parameter_tangent_matches_oracle():
    rng = np.random.default_rng(9)
    n, dim = 6, 2
    f0 = tf.constant(np.array([[0.9, 0.1], [0.0, 0.8]]), DTYPE)
    q = tf.constant(0.4 * np.eye(dim), DTYPE)
    h = tf.constant(np.eye(dim), DTYPE)
    r = tf.constant(0.6 * np.eye(dim), DTYPE)
    states = tf.constant(rng.standard_normal((n, dim)), DTYPE)
    covs = tf.constant(
        np.stack([np.eye(dim) * (0.5 + 0.2 * i) for i in range(n)]), DTYPE
    )
    noise = tf.constant(rng.standard_normal((n, dim)), DTYPE)
    observation = tf.constant(rng.standard_normal(dim), DTYPE)
    theta0 = tf.constant([1.0], DTYPE)

    def summary_fn(theta):
        value, _tangent, _ld, _dld = flow_value_and_parameter_tangent_lgssm(
            theta,
            f0,
            q,
            h,
            r,
            states,
            covs,
            noise,
            observation,
            substeps=16,
            with_tangent=False,
        )
        return tf.reduce_sum(tf.sin(value))  # nonlinear scalar summary

    oracle = oracle_forward_autodiff_score(summary_fn, theta0)

    value, tangent, _ld2, _dld2 = flow_value_and_parameter_tangent_lgssm(
        theta0,
        f0,
        q,
        h,
        r,
        states,
        covs,
        noise,
        observation,
        substeps=16,
        with_tangent=True,
    )
    analytical = tf.reduce_sum(tf.cos(value) * tangent[..., 0])
    err = abs(float(analytical.numpy()) - float(oracle[0].numpy()))
    scale = max(abs(float(oracle[0].numpy())), 1.0)
    assert err < 1.0e-4 * scale, (
        f"S3 analytical {float(analytical.numpy())} vs oracle "
        f"{float(oracle[0].numpy())}"
    )
