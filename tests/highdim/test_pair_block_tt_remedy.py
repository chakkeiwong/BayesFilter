"""Focused identities for the paired current/previous TT extension."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

from bayesfilter.highdim import observation_guided_tt_tf as obs
from bayesfilter.highdim import pair_block_tt_tf as pair
from bayesfilter.nonlinear.fixed_sgqf_tf import tf_standard_normal_ghq_level_rule
from bayesfilter.highdim.c2_gaussian_hermite_proposal_tf import _prefix_row_vectors

D = tf.float64


def test_conditional_mass_equals_expanded_coefficient_norm():
    """Independent integral: orthonormal coefficients have squared-norm mass."""
    cores = tuple(tf.random.stateless_normal(shape, [391, i], dtype=D)
                  for i, shape in enumerate(((1, 3, 3, 3), (3, 3, 3, 3), (3, 3, 3, 1))))
    v = tf.constant([[-.8, .3, 1.1], [1.2, -.2, .7], [.1, -.5, -.9]], D)
    conditional = pair.pair_conditional_cores(cores, v)
    coefficients = tf.ones([3, 1, 1], D)
    for c in conditional:
        expanded = tf.einsum('npa,nakb->npkb', coefficients, c)
        coefficients = tf.reshape(expanded, [3, -1, int(c.shape[-1])])
    exact_mass = tf.reduce_sum(tf.square(coefficients), axis=[1, 2])
    tf.debugging.assert_near(pair.conditional_normalizer_batched(conditional), exact_mass,
                             rtol=1e-12, atol=1e-12)
    compiled = tf.function(pair.conditional_normalizer_batched,
        input_signature=[tuple(tf.TensorSpec(c.shape, D) for c in conditional)], jit_compile=True)
    tf.debugging.assert_near(compiled(conditional), exact_mass, rtol=1e-12, atol=1e-12)


def test_zero_conditional_uses_gaussian_and_spectra_match_full_matrix():
    zeros = (tf.zeros([1, 2, 2, 1], D),)
    noise = tf.constant([[.3], [-.5]], D)
    x, logq, info = pair.sample_pair_conditional(zeros, tf.zeros([2, 1], D),
        tf.constant(.2, D), tf.constant([.2, .7], D), tf.constant([[.4], [.6]], D), noise)
    tf.debugging.assert_equal(x, noise)
    tf.debugging.assert_near(logq, -.5*noise[:, 0]**2-.5*tf.math.log(tf.constant(2*3.141592653589793, D)))
    assert bool(info['finite'])
    cores = _cores()
    matrix = tf.reshape(cores[0], [4, 2]) @ tf.reshape(cores[1], [2, 4])
    expected = tf.linalg.svd(matrix, compute_uv=False)[:2]
    actual = pair.pair_bond_spectra(cores)[0]
    tf.debugging.assert_near(actual['singular_values'], expected, atol=1e-12)
    assert int(actual['effective_rank']) == 2


def _cores():
    c1 = tf.constant([[[[1.0, .2], [0., .1]], [[0.3, -.1], [1.0, .2]]]], D)
    c2 = tf.constant([[[[1.0], [.2]], [[-.1], [.7]]],
                      [[[.4], [-.3]], [[.8], [.1]]]], D)
    return (c1, c2)


def test_pair_expansion_preserves_evaluation_and_mass():
    cores = _cores()
    rows = tf.random.stateless_normal([31, 2, 2], [19, 3], dtype=D)
    direct = pair.evaluate_pair_cores(cores, rows)
    scalar = _prefix_row_vectors(pair.pair_to_scalar_cores(cores),
                                       tf.reshape(rows, [31, 4]))[:, 0]
    tf.debugging.assert_near(direct, scalar, atol=2e-12)
    mass = pair.pair_total_mass(cores)
    rule = tf_standard_normal_ghq_level_rule(7)
    g0, g1, g2, g3 = tf.meshgrid(rule.nodes, rule.nodes, rule.nodes, rule.nodes, indexing="ij")
    points = tf.stack([g0, g1, g2, g3], axis=-1)
    points = tf.reshape(points, [-1, 2, 2])
    w0, w1, w2, w3 = tf.meshgrid(rule.weights, rule.weights, rule.weights, rule.weights, indexing="ij")
    weights = tf.reshape(w0*w1*w2*w3, [-1])
    values = pair.evaluate_pair_cores(cores, points)
    total = tf.reduce_sum(weights * values * values)
    tf.debugging.assert_near(mass, total, atol=3e-10)


def test_retained_marginal_and_particle_conditional_are_normalized():
    cores = _cores()
    rule = tf_standard_normal_ghq_level_rule(7)
    u = tf.constant([[-.4, .3], [.6, -.2]], D)
    q = pair.pair_retained_quadratic(cores, u)
    integrated = []
    for ui in u:
        value = tf.constant(0., D)
        for a in range(int(rule.nodes.shape[0])):
            for b in range(int(rule.nodes.shape[0])):
                row = tf.stack([[ui[0], rule.nodes[a]],
                                [ui[1], rule.nodes[b]]], axis=0)[None, :, :]
                h = pair.evaluate_pair_cores(cores, row)[0]
                value += rule.weights[a] * rule.weights[b] * h*h
        integrated.append(value)
    tf.debugging.assert_near(q, tf.stack(integrated), atol=3e-10)
    v = tf.constant([[-1.1, .2], [.8, -.7]], D)
    conditional = pair.pair_conditional_cores(cores, v)
    z = pair.conditional_normalizer_batched(conditional)
    assert float(tf.reduce_max(z)) > 0.
    sample = pair.sample_pair_conditional(cores, v, tf.constant(.1, D),
        tf.zeros([2], D), tf.constant([[.3, .7], [.6, .2]], D), tf.zeros([2, 2], D))
    assert bool(sample[2]["finite"])
    assert float(sample[2]["cdf_residual"]) < 2e-12
    assert float(tf.linalg.norm(sample[0][0] - sample[0][1])) > 1e-4


def test_joint_sgqf_mixture_has_bounded_importance_weights():
    model = obs.SVModel(tf.constant([[.6]], D), tf.constant([[1.5625]], D))
    current = obs.Chart.from_moments(tf.constant([.2], D), tf.constant([[1.]], D))
    previous = obs.Chart.from_moments(tf.constant([.1], D), tf.constant([[1.]], D))
    rows, logw, info = obs.joint_sgqf_row_sampler(model, current, previous, 128, 71)
    assert rows.shape == (128, 2)
    tf.debugging.assert_all_finite(logw, "importance log weights")
    assert float(info["maximum_rho_over_s"]) <= 5.0000001
    assert float(info["row_ess"]) > 1.


def test_pair_fit_reports_convergence_and_keeps_audit_separate():
    def log_target(row):
        u, v = row[:, :, 0], row[:, :, 1]
        h = 1. + u[:, 0]*v[:, 0] + .5*u[:, 1]*v[:, 1]
        return tf.math.log(tf.square(h) + .1)

    cores, info = obs.pair_fit_from_log_target(log_target, 2, 81, rows=64,
        degree=2, rank=2, sweeps=1, proximal_steps=8, penalty=0.)
    assert len(cores) == 2
    assert info["audit_used_for_selection"] is False
    assert float(info["fit"]["kkt_residual"]) >= 0.
    assert float(info["audit_relative_rms"]) < 2.
    reloaded = tuple(tf.io.parse_tensor(tf.io.serialize_tensor(x), out_type=D) for x in cores)
    tf.debugging.assert_near(pair.evaluate_pair_cores(cores,
        tf.random.stateless_normal([9, 2, 2], [8, 9], dtype=D)),
        pair.evaluate_pair_cores(reloaded,
        tf.random.stateless_normal([9, 2, 2], [8, 9], dtype=D)), atol=1e-12)


def test_conditional_inverse_matches_independent_quadratic_cdf():
    # Independent diagnostic polynomial algebra (degree one, d=2), not the
    # shared Hermite-CDF implementation being tested.
    import math
    import numpy as np
    cores = _cores()
    v = tf.constant([[-1.1, .2], [.8, -.7]], D)
    uniforms = tf.constant([[.3, .7], [.6, .2]], D)
    conditional = pair.pair_conditional_cores(cores, v)
    kr = pair.batched_inverse_kr(conditional, uniforms)
    for j in range(2):
        h = conditional[0][j, 0].numpy() @ conditional[1][j, :, :, 0].numpy()
        u = kr['reference_points'][j].numpy()
        coeffs = ((np.sum(h[1]**2), 2*np.dot(h[0], h[1]), np.sum(h[0]**2)),)
        b = np.array([1., u[0]]) @ h
        coeffs += ((b[1]**2, 2*b[0]*b[1], b[0]**2),)
        for axis, (a, b, c) in enumerate(coeffs):
            x = float(u[axis])
            cdf = .5*math.erfc(-x/math.sqrt(2)) - (a*x+b)*math.exp(-x*x/2)/math.sqrt(2*math.pi)/(a+c)
            assert abs(cdf-float(uniforms[j, axis])) < 2e-12


def test_defensive_row_sampler_preserves_reference_moments():
    model = obs.SVModel(tf.constant([[.6, .1], [0., .4]], D), tf.eye(2, dtype=D))
    current = obs.Chart.from_moments(tf.constant([.3, -.2], D), tf.constant([[.7, .1], [.1, .8]], D))
    previous = obs.Chart.from_moments(tf.constant([-.1, .2], D), tf.constant([[1., .1], [.1, .9]], D))
    rows, logw, _ = obs.joint_sgqf_row_sampler(model, current, previous, 65536, 771)
    w = tf.exp(logw)
    tf.debugging.assert_near(tf.reduce_mean(w), tf.constant(1., D), atol=.02)
    tf.debugging.assert_near(tf.reduce_mean(w[:, None]*rows**2, axis=0), tf.ones([4], D), atol=.06)
    assert abs(float(tf.reduce_mean(w*rows[:, 0]*rows[:, 2]))) < .05


def test_importance_rows_keep_grouped_coordinate_meaning():
    seen = []
    def sampler(count, seed):
        rows = tf.tile(tf.constant([[.1, .2, .7, .8]], D), [count, 1])
        return rows, tf.zeros([count], D), {}
    def log_target(row):
        seen.append(row[0])
        return tf.zeros([tf.shape(row)[0]], D)
    obs.pair_fit_from_log_target(log_target, 2, 17, rows=8, degree=1, rank=1,
        sweeps=1, proximal_steps=4, row_sampler=sampler, jit_compile=False)
    for row in seen:
        tf.debugging.assert_equal(row, tf.constant([[.1, .7], [.2, .8]], D))
