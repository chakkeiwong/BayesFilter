"""Independent law and analytical-derivative checks, deliberately CPU reference."""
import math

import pytest
import tensorflow as tf

from bayesfilter.wrapped_gaussian_tf import alias_policy, wrapped_gaussian_kernel


def test_declared_alias_bound_and_invalid_policy():
    p = alias_policy()
    assert p["omitted_mass_upper_bound"] <= 1e-14
    for sigma in (0., -1., math.inf, math.nan):
        with pytest.raises(ValueError):
            alias_policy(sigma)


def test_correlated_density_matches_independent_bivariate_alias_sum_and_tangent():
    x = tf.constant([[.2, -.9], [-2., 3.], [.7, 0.]], tf.float64)
    dx = tf.constant([[.1, -.3], [.2, .4], [-.5, .1]], tf.float64)
    covariance = tf.constant([[.6, .25], [.25, 2.]], tf.float64)
    dc = tf.constant([[.2, -.04], [-.04, .3]], tf.float64)
    kernel = wrapped_gaussian_kernel()
    out = kernel(x, covariance, dx, dc)
    assert bool(tf.reduce_all(out["value_valid"] & out["score_valid"]))
    shifts = tf.stack([tf.zeros([33], tf.float64), 2*math.pi*tf.cast(tf.range(-16, 17), tf.float64)], -1)
    residuals = x[:, None, :]+shifts[None, :, :]
    inverse = tf.linalg.inv(covariance)
    reference = tf.reduce_logsumexp(-.5*tf.einsum("nki,ij,nkj->nk", residuals, inverse, residuals)
        -math.log(2*math.pi)-.5*tf.linalg.logdet(covariance), axis=1)
    tf.debugging.assert_near(out["value"], reference, atol=2e-12, rtol=2e-12)
    eps = tf.constant(1e-5, tf.float64)
    plus = kernel(x+eps*dx, covariance+eps*dc, dx, dc)["value"]
    minus = kernel(x-eps*dx, covariance-eps*dc, dx, dc)["value"]
    tf.debugging.assert_near(out["tangent"], (plus-minus)/(2*eps), atol=2e-9, rtol=2e-9)


def test_periodicity_branch_boundary_and_scale_veto():
    kernel = wrapped_gaussian_kernel()
    x = tf.constant([[0., -math.pi], [0., math.pi], [.3, .4], [.3, .4+12*math.pi]], tf.float64)
    covariance = tf.constant([[1., 0.], [0., math.pi**2]], tf.float64)
    out = kernel(x, covariance, tf.ones_like(x), tf.zeros_like(covariance))
    tf.debugging.assert_near(out["value"][::2], out["value"][1::2], atol=1e-12, rtol=1e-12)
    tf.debugging.assert_near(out["tangent"][::2], out["tangent"][1::2], atol=1e-12, rtol=1e-12)
    for bad in (tf.constant([[1., 0.], [0., 0.]], tf.float64), covariance*2,
                tf.constant([[1., .1], [.2, 1.]], tf.float64)):
        result = kernel(x, bad, tf.zeros_like(x), tf.zeros_like(bad))
        assert not bool(tf.reduce_any(result["value_valid"]))
        assert not bool(tf.reduce_any(tf.math.is_finite(result["value"])))
