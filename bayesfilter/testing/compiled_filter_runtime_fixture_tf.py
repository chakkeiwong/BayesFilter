"""Nonlinear, non-diagonal multi-chain fixture for compiled filter diagnostics.

This is an engineering fixture, not a statistical target or DZ5 qualification.
"""

from __future__ import annotations

import tensorflow as tf

from bayesfilter.nonlinear.factor_srukf_tf import (
    TFFactorSRUKFDerivatives,
    TFFactorSRUKFModel,
    tf_factor_srukf_value_and_score,
)
from bayesfilter.nonlinear.rectangular_srukf_tf import (
    TFRectangularSRUKFDerivatives,
    TFRectangularSRUKFFixedBranch,
    TFRectangularSRUKFModel,
    tf_rectangular_srukf_value_and_score,
)


def fixture_components(theta, *, rectangular=False, joint=False):
    theta = tf.convert_to_tensor(theta, tf.float64)
    b = int(theta.shape[0])
    n, p = 2, 3
    mean = tf.stack((0.2 + 0.1 * theta[:, 2], -0.15 + 0.03 * theta[:, 2]), -1)
    initial = tf.broadcast_to(
        tf.constant([[0.7, 0.0], [0.13, 0.55]], tf.float64), [b, n, n]
    )
    process = (
        tf.exp(0.2 * theta[:, 1, None, None])
        * tf.constant([[0.23, 0.0], [0.04, 0.17]], tf.float64)[None]
    )
    obsfactor = tf.broadcast_to(
        tf.constant([[0.31, 0.0], [-0.04, 0.28]], tf.float64), [b, n, n]
    )
    dmean = tf.broadcast_to(
        tf.constant([[0.0, 0.0], [0.0, 0.0], [0.1, 0.03]], tf.float64), [b, p, n]
    )
    dinitial = tf.zeros([b, p, n, n], tf.float64)
    dprocess = (
        tf.one_hot(1, p, dtype=tf.float64)[None, :, None, None] * 0.2 * process[:, None]
    )
    dobsfactor = tf.zeros([b, p, n, n], tf.float64)

    def transition(x, q):
        return tf.stack(
            (
                (0.8 + 0.1 * theta[:, 0, None]) * x[..., 0]
                + 0.05 * x[..., 1] ** 2
                + q[..., 0],
                0.6 * x[..., 1] + 0.02 * tf.sin(x[..., 0]) + q[..., 1],
            ),
            -1,
        )

    def sj(x, q):
        del q
        a = tf.broadcast_to(0.8 + 0.1 * theta[:, 0, None], tf.shape(x)[:2])
        return tf.stack(
            (
                tf.stack((a, 0.1 * x[..., 1]), -1),
                tf.stack(
                    (
                        0.02 * tf.cos(x[..., 0]),
                        tf.fill(tf.shape(a), tf.constant(0.6, tf.float64)),
                    ),
                    -1,
                ),
            ),
            -2,
        )

    def qj(x, q):
        del q
        return tf.broadcast_to(tf.eye(n, dtype=tf.float64), [b, tf.shape(x)[1], n, n])

    def dt(x, q):
        del q
        direction = tf.stack((0.1 * x[..., 0], tf.zeros_like(x[..., 0])), -1)
        return (
            tf.one_hot(0, p, dtype=tf.float64)[None, :, None, None] * direction[:, None]
        )

    def observe(x):
        return tf.stack(
            (x[..., 0] + 0.1 * x[..., 1] ** 2, 0.5 * x[..., 1] + 0.03 * x[..., 0] ** 2),
            -1,
        )

    def oj(x):
        return tf.stack(
            (
                tf.stack((tf.ones_like(x[..., 0]), 0.2 * x[..., 1]), -1),
                tf.stack(
                    (
                        0.06 * x[..., 0],
                        tf.fill(tf.shape(x)[:2], tf.constant(0.5, tf.float64)),
                    ),
                    -1,
                ),
            ),
            -2,
        )

    def od(x):
        return tf.zeros([b, p, tf.shape(x)[1], n], tf.float64)

    def fused(x, q):
        return transition(x, q), sj(x, q), qj(x, q), dt(x, q)

    model_type = TFRectangularSRUKFModel if rectangular else TFFactorSRUKFModel
    derivatives_type = (
        TFRectangularSRUKFDerivatives if rectangular else TFFactorSRUKFDerivatives
    )
    model = model_type(mean, initial, process, obsfactor, transition, observe)
    derivatives = derivatives_type(
        dmean,
        dinitial,
        dprocess,
        dobsfactor,
        sj,
        qj,
        dt,
        oj,
        od,
        transition_value_and_derivatives_fn=fused if joint else None,
    )
    return model, derivatives


def fixture_result(
    theta, *, rectangular=False, joint=False, jit_compile=True, horizon=4
):
    b = int(theta.shape[0])
    model, derivatives = fixture_components(theta, rectangular=rectangular, joint=joint)
    dates = tf.cast(tf.range(horizon), tf.float64)
    y = tf.stack((0.25 + 0.04 * tf.sin(dates), -0.12 + 0.03 * tf.cos(dates)), -1)
    y = tf.broadcast_to(y[None], [b, horizon, 2])
    if rectangular:
        branch = TFRectangularSRUKFFixedBranch(2, (0, 1), 2, (0, 1), 2, (0, 1))
        return tf_rectangular_srukf_value_and_score(
            y, model, derivatives, branch=branch, jit_compile=jit_compile
        )
    return tf_factor_srukf_value_and_score(
        y, model, derivatives, jit_compile=jit_compile
    )


def numerical_result(result):
    return (
        result.log_likelihood,
        result.score,
        result.filtered_mean,
        result.filtered_factor,
        result.d_filtered_mean,
        result.d_filtered_factor,
    )


class CompiledFilterMechanicsAdapter:
    """Fixed dense affine map plus nonlinear map, with exact analytical score.

    The affine scale combines a frozen transport and non-identity diagonal mass
    whitening. This tests both coordinate factors inside the numerical block;
    it issues no tuning or admission artifact.
    """

    parameter_dim = 3

    def __init__(self, *, rectangular=True):
        self.rectangular = rectangular
        self.transport = tf.constant(
            [[1.1, 0.0, 0.0], [0.12, 0.85, 0.0], [0.04, -0.06, 1.2]], tf.float64
        )
        self.mass_scale = tf.constant([0.8, 1.1, 0.9], tf.float64)
        self.offset = tf.constant([0.12, -0.17, 0.23], tf.float64)

    def adapter_signature(self):
        return f"compiled-filter-mechanics-20260916-rectangular-{self.rectangular}"

    def value_score_capability(self):
        from bayesfilter.inference import ValueScoreCapability

        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            runtime_backend="tensorflow",
            evidence_path="docs/plans/kalman_ukf_compiled_runtime_repair_20260916.md",
            target_scope="compiled_filter_mechanics_diagnostic",
            nonclaims=(
                "short engineering fixture only; no posterior or tuning admission",
            ),
            full_chain_xla_diagnostic_ready=True,
        )

    def log_prob_and_grad(self, z):
        whitened = z * self.mass_scale
        affine = whitened @ tf.transpose(self.transport) + self.offset
        theta = affine + 0.03 * tf.tanh(affine)
        jacdiag = 1.0 + 0.03 * (1.0 - tf.tanh(affine) ** 2)
        result = fixture_result(theta, rectangular=self.rectangular, joint=True)
        value = result.log_likelihood - 0.5 * tf.reduce_sum(theta**2, -1)
        value += tf.reduce_sum(tf.math.log(jacdiag), -1)
        value += tf.reduce_sum(
            tf.math.log(tf.linalg.diag_part(self.transport) * self.mass_scale)
        )
        score = result.score - theta
        grad_affine = (
            score * jacdiag
            - 0.06 * tf.tanh(affine) * (1.0 - tf.tanh(affine) ** 2) / jacdiag
        )
        return value, (grad_affine @ self.transport) * self.mass_scale
