"""Fresh reset identities and derivative checks, without old LEDH results.

These are cloud-primitive engineering checks; they grant no LEDH admission.
"""

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import ledh_contract_e_reset_tf as reset


def fixture(batch, count):
    rng = np.random.default_rng(91726 + count)
    shape = (batch, count, 2)
    source = rng.normal(size=shape) @ np.array([[1.3, .2], [.1, .9]])
    weights = rng.uniform(.7, 1.3, size=(batch, count))
    weights /= weights.sum(axis=1, keepdims=True)
    transported = .15 * rng.normal(size=shape)
    residual = rng.normal(size=shape)
    residual -= residual.mean(axis=1, keepdims=True)
    ridge = np.full(batch, .03)
    inputs = tuple(tf.constant(value, tf.float64) for value in (source, weights, transported, residual, ridge))
    tangents = tuple(tf.constant(rng.normal(size=value.shape)*.03, tf.float64) for value in inputs)
    return inputs, tangents, tf.constant(rng.normal(size=shape), tf.float64)


def independent_reset(source, weights, transported, residual, ridge):
    mean = np.einsum("bn,bni->bi", weights, source)
    centered = source - mean[:, None]
    target_cov = np.einsum("bn,bni,bnj->bij", weights, centered, centered)
    plus = transported - transported.mean(axis=1, keepdims=True)
    plus_cov = np.einsum("bni,bnj->bij", plus, plus) / source.shape[1]
    regularizer = ridge[:, None, None] * np.eye(source.shape[-1])
    injection = np.linalg.cholesky(target_cov - plus_cov + regularizer)
    injected = transported + residual @ np.swapaxes(injection, -1, -2)
    centered_injected = injected - injected.mean(axis=1, keepdims=True)
    injected_cov = np.einsum("bni,bnj->bij", centered_injected, centered_injected) / source.shape[1]
    left, right = np.linalg.cholesky(target_cov+regularizer), np.linalg.cholesky(injected_cov+regularizer)
    affine = np.swapaxes(np.linalg.solve(np.swapaxes(right, -1, -2), np.swapaxes(left, -1, -2)), -1, -2)
    return mean[:, None] + centered_injected @ np.swapaxes(affine, -1, -2)


@pytest.mark.parametrize("batch,count", [(1, 8), (2, 12)])
def test_cloud_reset_primal_and_total_derivatives(batch, count):
    inputs, tangents, upstream = fixture(batch, count)
    forward = reset.contract_e_chol_cloud_forward_tf(*inputs)
    expected = independent_reset(*(value.numpy() for value in inputs))
    np.testing.assert_allclose(forward["particles"], expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(forward["mean_residual"], 0, atol=1e-12)
    np.testing.assert_allclose(forward["ridged_identity_residual"], 0, atol=1e-12)
    np.testing.assert_allclose(forward["raw_covariance_residual"], forward["predicted_raw_covariance_residual"], atol=1e-12)

    jvp = reset.contract_e_chol_cloud_jvp_tf(*inputs, *tangents)
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        particles = reset._contract_e_chol_cloud_forward_core(*inputs)["particles"]
        objective = tf.reduce_sum(particles * upstream)
    expected_adjoint = tape.gradient(objective, inputs)
    adjoints = reset.contract_e_chol_cloud_vjp_tf(*inputs, upstream)
    keys = ("source_particles", "normalized_weights", "transported_particles", "residual_design", "ridge")
    for key, expected in zip(keys, expected_adjoint, strict=True):
        np.testing.assert_allclose(adjoints[key], expected, rtol=1e-11, atol=1e-11)
    dual = tf.add_n([tf.reduce_sum(adjoints[key]*direction) for key, direction in zip(keys, tangents, strict=True)])
    np.testing.assert_allclose(tf.reduce_sum(jvp*upstream), dual, rtol=1e-11, atol=1e-11)
    epsilon = 1e-5
    plus = independent_reset(*(value.numpy()+epsilon*direction.numpy() for value, direction in zip(inputs, tangents, strict=True)))
    minus = independent_reset(*(value.numpy()-epsilon*direction.numpy() for value, direction in zip(inputs, tangents, strict=True)))
    np.testing.assert_allclose(jvp, (plus-minus)/(2*epsilon), rtol=2e-7, atol=2e-8)

    @tf.function(input_signature=[tf.TensorSpec(value.shape, value.dtype) for value in inputs], jit_compile=True, autograph=False)
    def enclosing(*values):
        return reset.contract_e_chol_cloud_forward_tf(*values)["particles"]

    np.testing.assert_allclose(enclosing(*inputs), forward["particles"], rtol=1e-12, atol=1e-12)
    assert "HloModule" in enclosing.experimental_get_compiler_ir(*inputs)(stage="optimized_hlo")
    assert reset.contract_e_chol_cloud_forward_tf._jit_compile is True
