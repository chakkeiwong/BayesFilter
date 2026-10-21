"""CPU-only independent scalar and rejection checks for stationary initialization."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import pytest
import tensorflow as tf
from bayesfilter.highdim.ledh_linear_parameter_adapter_tf import stationary_covariance_and_tangent

@pytest.mark.parametrize("dtype",[tf.float32,tf.float64])
def test_scalar_covariance_and_total_directional_derivative(dtype):
    f,q,df,dq=(tf.constant([[x]],dtype) for x in (.7,1.4,.2,.3))
    cov,tangent,status=stationary_covariance_and_tangent(f,q,df,dq)
    f64,q64,df64,dq64=(tf.cast(x,tf.float64) for x in (f,q,df,dq))
    exact=q64/(1-f64*f64)
    derivative=dq64/(1-f64*f64)+2*f64*df64*q64/(1-f64*f64)**2
    tolerance=1e-6 if dtype==tf.float32 else 1e-13
    tf.debugging.assert_near(tf.cast(cov,tf.float64),exact,atol=tolerance,rtol=tolerance)
    tf.debugging.assert_near(tf.cast(tangent,tf.float64),derivative,atol=tolerance,rtol=tolerance)
    assert bool(status["value_valid"]) and bool(status["force_valid"])
    assert cov.dtype==dtype and tangent.dtype==dtype

@pytest.mark.parametrize("transition", [[[1.]], [[1.1]], [[.5,2.],[0.,.5]]])
def test_uncertified_dynamics_are_rejected(transition):
    f=tf.constant(transition,tf.float32)
    q=tf.eye(f.shape[0],dtype=f.dtype)
    cov,tangent,status=stationary_covariance_and_tangent(f,q,tf.zeros_like(f),tf.zeros_like(q))
    assert not bool(status["value_valid"]) and not bool(status["force_valid"])
    assert bool(tf.reduce_all(tf.math.is_nan(cov)))
    assert bool(tf.reduce_all(tf.math.is_nan(tangent)))

def test_nonfinite_derivative_does_not_invalidate_finite_value():
    cov,tangent,status=stationary_covariance_and_tangent(
        tf.constant([[.5]]),tf.constant([[1.]]),tf.constant([[float("nan")]]),tf.constant([[0.]]))
    assert bool(status["value_valid"]) and not bool(status["force_valid"])
    assert bool(tf.reduce_all(tf.math.is_finite(cov)))
    assert bool(tf.reduce_all(tf.math.is_nan(tangent)))
