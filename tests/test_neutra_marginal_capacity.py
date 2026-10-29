"""Constructive capacity and derivative checks; no training-quality claim."""
import math

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
from bayesfilter.inference.neutra_transport import NeuTraTransport, NeuTraTransportConfig


def witness(kind, stages=1, permutation="full_reverse"):
    # Tiny hand-constructed witnesses, not architecture or training defaults.
    flow = NeuTraTransport(NeuTraTransportConfig(2, kind, (2,), stages, "tanh",
        (247, 19), 2., mixture_components=2, permutation_policy=permutation,
        naf_conditioner="paper_made"))
    for stage in flow.stages:
        for value in stage.trainable_variables:
            value.assign(tf.zeros_like(value))
        if kind == "iaf":
            # y1=z1, y2=z2+tanh(z1). Reverse then compose permits nonlinear y1.
            stage.weights[0].assign([[1., 0.], [0., 0.]])
            stage.weights[-1].assign([[0., 0., 0., 1.], [0., 0., 0., 0.]])
        else:
            raw = math.log(math.expm1(1.-flow.config.slope_floor))
            # Equal unit slopes, different offsets: a nonlinear monotone
            # unconditional first-coordinate DSF, despite its constant conditioner.
            stage.biases[-1].assign([raw, raw, raw, raw, -2., -2., 2., 2., 0., 0., 0., 0.])
    return flow


def test_legacy_first_marginal_restriction_and_permuted_repair():
    z = tf.constant([[-2., 0.], [-1., 0.], [0., 0.], [1., 0.], [2., 0.]], tf.float64)
    single = witness("iaf")
    preserving = witness("iaf", stages=2, permutation="root_preserving_reverse")
    permuted = witness("iaf", stages=2)
    np.testing.assert_allclose(single.forward_batch(z)[:, 0], z[:, 0], atol=1e-12)
    np.testing.assert_allclose(preserving.forward_batch(z)[:, 0], z[:, 0], atol=1e-12)
    first = permuted.forward_batch(z)[:, 0]
    np.testing.assert_allclose(first, tf.math.tanh(z[:, 0]), atol=1e-12)
    assert abs(float(first[4]-2*first[3]+first[2])) > .1


@pytest.mark.parametrize("kind,stages", [("iaf", 2), ("naf_dsf", 1)])
def test_nonlinear_capacity_frozen_inverse_jacobian_and_score(kind, stages):
    flow = witness(kind, stages)
    z = tf.constant([[-2., .2], [-1., -.3], [0., .7], [1., -.5], [2., .3]], tf.float64)
    signature = "a" * 64
    frozen = load_frozen_neutra_artifact(flow.frozen_payload(target_signature=signature),
        expected_target_signature=signature).transport

    @tf.function(input_signature=[tf.TensorSpec([5, 2], tf.float64)], jit_compile=True)
    def evaluate(values):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(values)
            y, ld = frozen.forward_and_logdet(values)
            target = -.5*tf.reduce_sum(tf.square(y), axis=-1)+ld
            components = [y[:, i] for i in range(2)]
        # Explicit reverse passes; no pfor or row-mapped target evaluation.
        jacobian = tf.stack([tape.gradient(c, values) for c in components], axis=1)
        automatic_score = tape.gradient(target, values)
        score = frozen.pullback_score_batch(values, -y)+frozen.log_abs_det_jacobian_score_batch(values)
        recovered, recovered_ld = frozen.inverse_and_forward_logdet(y)
        return y, ld, jacobian, score, automatic_score, recovered, recovered_ld

    y, ld, jacobian, score, automatic, recovered, recovered_ld = evaluate(z)
    np.testing.assert_array_equal(frozen.forward_batch(z), flow.forward_batch(z))
    # XLA and eager arithmetic can differ by FP64 rounding.
    np.testing.assert_allclose(y, flow.forward_batch(z), atol=1e-12, rtol=1e-12)
    # Existing FP64 inverse contract is 1e-11; allow accumulated inversion error.
    np.testing.assert_allclose(recovered, z, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(recovered_ld, ld, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(ld, tf.linalg.slogdet(jacobian)[1], atol=1e-10)
    np.testing.assert_allclose(score, automatic, atol=1e-10)
    if kind == "naf_dsf":
        assert abs(float(y[4, 0]-2*y[3, 0]+y[2, 0])) > .01
    assert evaluate.experimental_get_tracing_count() == 1
