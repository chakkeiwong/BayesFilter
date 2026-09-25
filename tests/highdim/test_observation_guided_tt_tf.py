"""Independent density identities and consumer wiring for observation-guided TT."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import tensorflow as tf
import pytest

from bayesfilter.highdim import observation_guided_tt_tf as m
from bayesfilter.highdim.c2_gaussian_hermite_proposal_tf import (
    normalized_hermite_incomplete_gram, _conditional_cdf,
    _normalized_hermite_values, _prefix_row_vectors, _paired_right_environments,
)
from bayesfilter.nonlinear.fixed_sgqf_tf import tf_standard_normal_ghq_level_rule, tf_fixed_sgqf_cloud
D = tf.float64


def model():
    return m.SVModel(tf.constant([[.6]], D), tf.constant([[1/.64]], D))


def step():
    # Nonuniform, nonseparable, two current coordinates and one fixed suffix.
    cores = tuple(tf.random.stateless_normal(shape, [431, k], dtype=D) * .3
                  for k, shape in enumerate(((1, 3, 2), (2, 3, 2), (2, 3, 1))))
    chart = m.Chart(tf.constant([.2, -.4], D), tf.constant([[1.3, 0.], [.2, .8]], D))
    return m.TTStep(cores, chart, m.Chart(tf.zeros([1], D), tf.ones([1, 1], D)),
                    tf.constant(.07, D), {}, 1)


def test_upper_conditional_density_normalizes_and_changes_with_suffix():
    s = step()
    rule = tf_standard_normal_ghq_level_rule(5)
    a, b = tf.meshgrid(rule.nodes, rule.nodes, indexing="ij")
    points = tf.stack([tf.reshape(a, [-1]), tf.reshape(b, [-1])], axis=1)
    weights = tf.reshape(rule.weights[:, None]*rule.weights[None, :], [-1])
    v = tf.constant([[-1.1], [.8]], D)
    suffix = m.conditional_suffix_environment(s.cores[2:], v)
    left = tf.ones([1, 1], D)
    for core in s.cores[:2]:
        left = tf.einsum("ac,akb,ckd->bd", left, core, core)
    z = tf.einsum("ab,nab->n", left, suffix)
    vector = _prefix_row_vectors(s.cores[:2], points)
    qratio = (tf.einsum("na,pab,nb->pn", vector, suffix, vector)+s.tau)/(z[:, None]+s.tau)
    tf.debugging.assert_near(tf.linalg.matvec(qratio, weights), tf.ones([2], D), atol=2e-12)
    sampler = m.compiled_conditional_sampler(s, False)
    u = tf.constant([[.3, .7], [.3, .7]], D)
    draws, logq, residual, valid, finite, _ = sampler(s.cores, s.tau, v, tf.zeros([2], D), u, tf.zeros([2, 2], D))
    assert bool(valid & finite)
    assert float(residual) < 2e-12
    assert float(tf.linalg.norm(draws[0]-draws[1])) > .01
    vector = _prefix_row_vectors(s.cores[:2], draws)
    expected = tf.math.log(tf.einsum("na,nab,nb->n", vector, suffix, vector)+s.tau) + m._log_standard_normal(draws)-tf.math.log(z+s.tau)
    tf.debugging.assert_near(logq, expected, atol=2e-12)
    # The first generated coordinate is current-axis 1, conditioned on suffix.
    left_integral = tf.einsum("akb,akd->bd", s.cores[0], s.cores[0])
    core_reverse = tf.transpose(s.cores[1], [2, 1, 0])
    cdf = _conditional_cdf(suffix, core_reverse, left_integral, draws[:, 1], 2, z)
    tf.debugging.assert_near(cdf, u[:, 1], atol=2e-12)


def test_incomplete_gram_derivative_is_actual_polynomial_density():
    x = tf.constant([-.7, .2, 1.4], D)
    coeff = tf.constant([.3, -.5, .8, .1], D)
    with tf.GradientTape() as tape:
        tape.watch(x)
        cdf = tf.einsum("a,nab,b->n", coeff, normalized_hermite_incomplete_gram(x, 3), coeff)
    derivative = tape.gradient(cdf, x)
    density = tf.square(tf.linalg.matvec(_normalized_hermite_values(x, 3), coeff)) * tf.exp(-x*x/2)/tf.sqrt(tf.constant(2*3.141592653589793, D))
    tf.debugging.assert_near(derivative, density, atol=2e-12)


def test_sgqf_updates_magnitude_and_history_with_valid_moments():
    mod = model()
    clouds = [(k, tf_fixed_sgqf_cloud(1, k)) for k in (4, 5)]
    predictive = m.Chart.from_moments(tf.zeros([1], D), mod.covariance0)
    small, _ = m.sgqf_update(mod, predictive, tf.constant([.05], D), clouds)
    large, _ = m.sgqf_update(mod, predictive, tf.constant([1.2], D), clouds)
    assert float(large.mean[0]-small.mean[0]) > .5
    a, _ = m.build_guide_path(mod, tf.constant([[.05], [.3]], D), levels=(4, 5))
    b, _ = m.build_guide_path(mod, tf.constant([[1.2], [.3]], D), levels=(4, 5))
    assert abs(float(a[1][1].mean[0]-b[1][1].mean[0])) > .1
    tf.debugging.assert_near(predictive.inverse(predictive.forward(tf.constant([[-1.], [.3]], D))), tf.constant([[-1.], [.3]], D), atol=1e-14)
    with pytest.raises(ValueError, match="SPD"):
        m.Chart.from_moments(tf.zeros([1], D), tf.constant([[-1e-10]], D))


def test_joint_fit_retained_marginal_and_sampler_are_wired(monkeypatch):
    mod = model()
    obs = tf.constant([[.3], [.1]], D)
    guide, _ = m.build_guide_path(mod, obs, levels=(4, 5))
    called = []
    original = m.inverse_hermite_polynomial_kr
    def observed(*args, **kwargs):
        called.append(kwargs.get("reverse"))
        return original(*args, **kwargs)
    monkeypatch.setattr(m, "inverse_hermite_polynomial_kr", observed)
    m._SAMPLERS.clear()
    path = m.build_tt_path(mod, obs, guide, guided=True, rows=96, rank=2, degree=2, sweeps=2, jit_compile=False)
    assert path[1].conditioning_chart is guide[0][1]
    assert len(path[1].cores) == 2
    assert path[1].fit_diagnostics["audit_used_for_selection"] is False
    initial, logq, _ = m.sample_tt_step(path[0], tf.zeros([64, 1], D), 72, False)
    draws, density, _ = m.sample_tt_step(path[1], initial, 73, False)
    assert called and all(called)
    assert bool(tf.reduce_all(tf.math.is_finite(draws + density[:, None])))
    # Exact marginal density vs independent integration of the full joint.
    rule = tf_standard_normal_ghq_level_rule(4)
    u = tf.constant([[-.4], [.6]], D)
    v = tf.tile(rule.nodes[:, None], [2, 1])
    uv = tf.concat([tf.repeat(u, len(rule.nodes), axis=0), v], axis=1)
    joint_poly = _prefix_row_vectors(path[1].cores, uv)[:, 0] ** 2
    integral = tf.linalg.matvec(tf.reshape(joint_poly, [2, -1]), rule.weights)
    retained = path[1].retained()
    tf.debugging.assert_near(integral, retained.reference_quadratic_form(u), atol=2e-10)
