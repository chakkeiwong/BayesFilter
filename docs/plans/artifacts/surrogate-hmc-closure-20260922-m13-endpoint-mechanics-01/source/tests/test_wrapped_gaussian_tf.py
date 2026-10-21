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


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
def test_batched_covariance_uses_shared_core_and_all_row_derivatives(dtype):
    from bayesfilter.wrapped_gaussian_tf import wrapped_gaussian_batch_kernel
    residual = tf.constant([[.2,-.9],[-.7,2.9],[.3,.4]],dtype)
    direction = tf.constant([[.1,-.3],[.2,.4],[-.5,.1]],dtype)
    covariance = tf.constant([[[.6,.25],[.25,2.]],[[.9,-.1],[-.1,.8]],[[.2,.03],[.03,1.2]]],dtype)
    dc = tf.constant([[[.2,-.04],[-.04,.3]],[[-.1,.02],[.02,.1]],[[.03,-.01],[-.01,-.2]]],dtype)
    batch = wrapped_gaussian_batch_kernel(dtype=dtype)
    shared = wrapped_gaussian_kernel(dtype=dtype)
    result = batch(residual,covariance,direction,dc)
    expected = [shared(residual[i:i+1],covariance[i],direction[i:i+1],dc[i]) for i in range(3)]
    tolerance = 3e-6 if dtype == tf.float32 else 1e-13
    for field in ("value","tangent"):
        assert result[field].numpy().tolist() == pytest.approx(
            [float(row[field][0].numpy()) for row in expected],rel=tolerance,abs=tolerance)
    h = 1e-3 if dtype == tf.float32 else 1e-6
    plus = batch(residual+h*direction,covariance+h*dc,tf.zeros_like(direction),tf.zeros_like(dc))["value"]
    minus = batch(residual-h*direction,covariance-h*dc,tf.zeros_like(direction),tf.zeros_like(dc))["value"]
    derivative = ((plus-minus)/(2*h)).numpy().tolist()
    assert result["tangent"].numpy().tolist() == pytest.approx(derivative,
        rel=2e-3 if dtype == tf.float32 else 2e-7,
        abs=2e-3 if dtype == tf.float32 else 2e-7)
    assert bool(tf.reduce_all(result["value_valid"] & result["score_valid"]).numpy())


def test_invalid_batched_covariance_rejects_only_its_own_row():
    from bayesfilter.wrapped_gaussian_tf import wrapped_gaussian_batch_kernel
    residual = tf.constant([[.1,.2],[.1,.2]],tf.float64)
    covariance = tf.constant([[[1.,0.],[0.,1.]],[[1.,0.],[0.,-1.]]],tf.float64)
    result = wrapped_gaussian_batch_kernel()(residual,covariance,
        tf.zeros_like(residual),tf.zeros_like(covariance))
    assert result["value_valid"].numpy().tolist() == [True,False]
    assert result["score_valid"].numpy().tolist() == [True,False]
