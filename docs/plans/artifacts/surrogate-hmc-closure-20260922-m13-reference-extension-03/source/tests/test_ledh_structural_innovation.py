"""CPU reference checks of the actual structural canonical call chain."""
from dataclasses import replace

import pytest
import tensorflow as tf

from bayesfilter.highdim.ledh_structural_innovation_tf import affine_innovation_geometry
from bayesfilter.testing.surrogate_hmc_structural_integration import fixture, batch_kernel, full_rank_pair


def test_geometry_rectangular_tangent_and_completion():
    b = tf.constant([[1., .1], [.2, 1.], [.3, -.4]], tf.float64)
    db = tf.constant([[.2, -.1], [.1, .1], [-.2, .3]], tf.float64)
    g = affine_innovation_geometry(b, db, state_dim=3, innovation_dim=2)
    eps = tf.constant(1e-5, tf.float64)
    gp = affine_innovation_geometry(b+eps*db, None, state_dim=3, innovation_dim=2)
    gm = affine_innovation_geometry(b-eps*db, None, state_dim=3, innovation_dim=2)
    tf.debugging.assert_near(g.left_inverse @ b, tf.eye(2, dtype=tf.float64), atol=1e-13)
    tf.debugging.assert_near(g.left_inverse_tangent, (gp.left_inverse-gm.left_inverse)/(2*eps), atol=1e-9)
    z = tf.constant([[.3, -.4], [.1, .7]], tf.float64)
    dz, a, da = z/5, tf.ones([2, 3], tf.float64), tf.ones([2, 3], tf.float64)/3
    x, dx = g.complete(z, dz, a, da)
    z2, dz2 = g.coordinates(x, dx, a, da)
    tf.debugging.assert_near(z, z2, atol=1e-13)
    tf.debugging.assert_near(dz, dz2, atol=1e-13)


def test_rank_deficient_basis_is_invalid_without_ridge():
    g = affine_innovation_geometry(tf.constant([[1., 1.], [0., 0.]], tf.float64),
                                  None, state_dim=2, innovation_dim=2)
    assert not bool(g.value_valid)
    assert not bool(g.force_valid)


def test_parameter_basis_requires_tangent_partner():
    model, _, _, _ = fixture()
    with pytest.raises(ValueError, match="supplied together"):
        replace(model, innovation_matrix_tangent_fn=None)


def test_structural_batch_scores_include_basis_noise_and_recursive_dependence():
    model, args, controls, theta = fixture(nonlinear=True)
    evaluate = batch_kernel(model, args, controls)
    directions = tf.eye(4, dtype=tf.float64)
    _, score, status = evaluate(tf.repeat(theta[None], 4, 0), directions)
    assert bool(tf.reduce_all(status["value_valid"] & status["force_valid"]))
    eps = tf.constant(1e-5, tf.float64)
    points = tf.concat([theta[None]+eps*directions, theta[None]-eps*directions], 0)
    values, _, checked = evaluate(points, tf.zeros_like(points))
    assert bool(tf.reduce_all(checked["value_valid"]))
    tf.debugging.assert_near(score, (values[:4]-values[4:])/(2*eps), atol=2e-5, rtol=2e-5)


def test_identity_chart_matches_existing_full_rank_program():
    plain, structural, args, controls, theta = full_rank_pair()
    direction = tf.constant([[1., -.2, 0., 0.]], tf.float64)
    a = batch_kernel(plain, args, controls)(theta[None], direction)
    b = batch_kernel(structural, args, controls)(theta[None], direction)
    for result in (a, b):
        assert bool(tf.reduce_all(result[2]["force_valid"] & result[2]["value_valid"]))
    tf.debugging.assert_near(a[0], b[0], atol=2e-8, rtol=2e-8)
    tf.debugging.assert_near(a[1], b[1], atol=2e-8, rtol=2e-8)


def test_zero_basis_rejection_reaches_actual_consumer():
    model, args, controls, theta = fixture(horizon=1)
    model = replace(model, innovation_matrix_fn=None, innovation_matrix_tangent_fn=None,
                    innovation_matrix=tf.zeros([2, 1], tf.float64))
    value, score, status = batch_kernel(model, args, controls)(theta[None], tf.ones([1, 4], tf.float64))
    assert not bool(status["value_valid"][0])
    assert not bool(status["force_valid"][0])
    assert bool(tf.math.is_inf(value[0]))
