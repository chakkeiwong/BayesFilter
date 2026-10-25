"""Independent, CPU-only diagnostics for the structural innovation adapter."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import math
import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim.analytical_jet_tf import Jet
from bayesfilter.highdim.covariance_proposal_score_batch_tf import make_batched_score_filter
from bayesfilter.highdim.covariance_proposal_tf import ProposalControls
from bayesfilter.highdim.structural_completion_proposal_tf import structural_proposal
from bayesfilter.testing.structural_ledh_models_tf import (
    StructuralModelSpec, make_model_a_kalman, make_structural_data,
    make_structural_design,
)

D = tf.float64


@pytest.mark.parametrize("name", ["model_a", "model_b"])
def test_completion_total_derivatives_and_innovation_jacobian(name):
    spec = StructuralModelSpec(name)
    q = spec.truth(D)[None]
    law = spec.structural_model(q)
    parent = law.q.constant(tf.constant([[[.3, -.2], [-.6, .4]]], D))
    parent.t = tf.random.stateless_normal(parent.t.shape, [101, 2], dtype=D) * .1
    e = law.q.constant(tf.constant([[[.8], [-.7]]], D))
    e.t = tf.random.stateless_normal(e.t.shape, [101, 3], dtype=D) * .1
    child = law.complete(parent, e)
    h = 1e-5
    for k in range(spec.parameter_count):
        direction = tf.one_hot(k, spec.parameter_count, dtype=D)[None]
        plus = spec.structural_model(q + h * direction)
        minus = spec.structural_model(q - h * direction)
        vp = plus.complete(plus.q.constant(parent.v + h * parent.t[k]),
                           plus.q.constant(e.v + h * e.t[k])).v
        vm = minus.complete(minus.q.constant(parent.v - h * parent.t[k]),
                            minus.q.constant(e.v - h * e.t[k])).v
        np.testing.assert_allclose(child.t[k], (vp - vm) / (2*h), rtol=1e-7, atol=1e-9)
    if name == "model_a":
        np.testing.assert_allclose(child.v[..., 1], parent.v[..., 0], atol=0)
    else:
        expected = .55 * parent.v[..., 1] + .8 * tf.tanh(child.v[..., 0])
        np.testing.assert_allclose(child.v[..., 1], expected, atol=1e-15)
    vp = law.observe(law.complete(parent, law.q.constant(e.v + h))).v
    vm = law.observe(law.complete(parent, law.q.constant(e.v - h))).v
    jac = law.innovation_observation_jacobian(parent, e)
    np.testing.assert_allclose(jac.v[..., 0], (vp-vm)/(2*h), rtol=1e-8, atol=1e-10)


def test_kalman_against_independent_singular_process_recursion():
    spec = StructuralModelSpec("model_a")
    data = make_structural_data(spec, 10, 341, D)
    program = make_model_a_kalman(1, 10, D, jit_compile=False)
    q = spec.truth(D)[None]
    result = program(q, data["observations"])
    f = np.array([[.35, -.10], [1., 0.]])
    m, p, total = np.zeros(2), np.eye(2), 0.
    for y in data["observations"].numpy()[:, 0]:
        m, p = f @ m, f @ p @ f.T + np.diag([.25**2, 0.])
        s, residual = p[0, 0]+.15**2, y-m[0]
        total += -.5*(math.log(2*math.pi*s)+residual**2/s)
        gain = p[:, 0]/s
        m, p = m+gain*residual, p-np.outer(gain, p[0])
    np.testing.assert_allclose(result["value"], [total], atol=1e-12)
    np.testing.assert_allclose(result["mean"][0], m, atol=1e-12)
    for k in range(2):
        dq = tf.one_hot(k, 2, dtype=D)[None]*1e-5
        fd = (program(q+dq, data["observations"])["value"]-
              program(q-dq, data["observations"])["value"])/2e-5
        np.testing.assert_allclose(result["score"][:, k], fd, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("name", ["model_a", "model_b"])
def test_conditional_prior_proposal_has_exact_structural_support(name):
    spec = StructuralModelSpec(name)
    law = spec.structural_model(spec.truth(D)[None])
    design = make_structural_design(16, 1, 931, D)
    x = law.initial_cloud(design["initial"])
    # Prior-only proposal: the importance correction is exactly g(y|child).
    beta = tf.constant([1., 0., 0.], D)
    indices = (tf.zeros([16], tf.int32), tf.range(16, dtype=tf.int32))
    child, densities, valid, maps = structural_proposal(
        law, x, tf.constant([.1], D), ProposalControls(flow_steps=8, reset_steps=5),
        beta, design["uniforms"][0], design["innovations"][0], indices=indices)
    expected = law.complete(x, law.q.constant(design["innovations"][0][None]))
    np.testing.assert_allclose(child.v, expected.v, atol=1e-12)
    normal = -.5*(math.log(2*math.pi)+design["innovations"][0, :, 0]**2)
    np.testing.assert_allclose(densities.v[0, :, 0], normal, atol=1e-12)
    assert bool(tf.reduce_all(valid)) and bool(tf.reduce_all(maps[6]))


@pytest.mark.parametrize("name", ["model_a", "model_b"])
def test_complete_filter_total_score_and_batch_parity(name):
    spec = StructuralModelSpec(name)
    n, horizon = 16, 2
    design = make_structural_design(n, horizon, 820, D)
    data = make_structural_data(spec, horizon, 821, D)
    controls = ProposalControls(flow_steps=8, reset_steps=5, correction_steps=1)
    kwargs = dict(spec=spec, n=n, horizon=horizon, controls=controls, dtype=D,
                  jit_compile=False, fixed_draw_design=(design["beta"], design["uniforms"]))
    batch = make_batched_score_filter(batch_size=2, **kwargs)
    scalar = make_batched_score_filter(batch_size=1, **kwargs)
    q = tf.stack([spec.truth(D), spec.truth(D)+.03])
    args = (data["observations"], design["initial"], design["innovations"])
    result = batch(q, *args)
    assert result["valid"].numpy().tolist() == [True, True]
    assert result["steps_completed"].numpy().tolist() == [horizon, horizon]
    for row in range(2):
        one = scalar(q[row:row+1], *args)
        for field in ("value", "score", "particles", "particle_tangent"):
            # Public output moves the batch axis first, including tangents.
            actual = result[field][row:row+1]
            np.testing.assert_allclose(actual, one[field], atol=2e-9, rtol=2e-8)
    for k in range(spec.parameter_count):
        dq = tf.one_hot(k, spec.parameter_count, dtype=D)[None]*2e-5
        fd = (batch(q+dq, *args)["value"]-batch(q-dq, *args)["value"])/4e-5
        np.testing.assert_allclose(result["score"][:, k], fd, atol=3e-6, rtol=3e-5)
