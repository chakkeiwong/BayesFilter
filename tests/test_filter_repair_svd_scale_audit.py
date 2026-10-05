"""Explanatory scale audit of actual Kalman telemetry and SRUKF SVD endpoints."""

import numpy as np
import tensorflow as tf

from bayesfilter.linear.kalman_tf import _spectral_telemetry
from bayesfilter.linear.rectangular_factor_tf import (
    batched_direct_stack_svd_factor,
    batched_direct_support_conditional,
)
from tests.test_filter_repair_geometry_control import clean, save

D = tf.float64


def test_actual_svd_endpoints_at_equal_condition_and_different_scale(request):
    @tf.function(input_signature=[tf.TensorSpec([1, 3, 3], D)], jit_compile=True, autograph=False)
    def telemetry(matrix):
        return _spectral_telemetry(matrix, tf.constant([True]))

    @tf.function(input_signature=[tf.TensorSpec([1, 3, 4], D)], jit_compile=True, autograph=False)
    def factor(matrix):
        result, singular, _, diagnostics = batched_direct_stack_svd_factor(matrix)
        return result, singular, diagnostics['rank']

    @tf.function(input_signature=[tf.TensorSpec([1, 3, 4], D), tf.TensorSpec([1, 2, 4], D),
        tf.TensorSpec([1, 3], D)], jit_compile=True, autograph=False)
    def conditional(observed, state, innovation):
        value, gain, result, rank, diagnostics = batched_direct_support_conditional(observed, state, innovation)
        return value, gain, result, rank, diagnostics['on_support']

    matrix = np.array([[2., 1., -.5], [1., 3., .25], [-.5, .25, 1.]])
    observed = np.concatenate([matrix, np.array([[.2], [.4], [.1]])], axis=1)
    state = np.array([[.1, .4, .3, .5], [1., .1, .2, .8]])
    records = []
    for scale in (1e-12, 1e-6, 1., 1e6):
        a, y, x = matrix * scale, observed * scale, state * scale
        innovation = y @ np.array([.1, .2, .3, -.1])
        minimum, condition = telemetry(tf.constant(a[None], D))
        computed, singular, rank = factor(tf.constant(y[None], D))
        value, gain, residual, conditional_rank, on_support = conditional(
            tf.constant(y[None], D), tf.constant(x[None], D), tf.constant(innovation[None], D))
        u, s, vh = np.linalg.svd(y, full_matrices=False)
        ref_gain = (x @ vh.T / s) @ u.T
        ref_residual = x - (x @ vh.T) @ vh
        coordinates = u.T @ innovation
        ref_value = -.5 * (3. * np.log(2. * np.pi) + 2. * np.log(s).sum() + np.square(coordinates / s).sum())
        factor_covariance = computed.numpy()[0] @ computed.numpy()[0].T / scale**2
        residual_covariance = residual.numpy()[0] @ residual.numpy()[0].T / scale**2
        ref_covariance = observed @ observed.T
        ref_residual_covariance = ref_residual @ ref_residual.T / scale**2
        records.append({'scale': scale, 'telemetry': clean((minimum, condition)),
            'reference_telemetry': [float(np.linalg.eigvalsh(a)[0]), float(np.linalg.cond(a))],
            'factor_rank': clean(rank), 'factor_singular': clean(singular), 'reference_singular': clean(s),
            'factor_relative_covariance_error': float(np.linalg.norm(factor_covariance-ref_covariance)/np.linalg.norm(ref_covariance)),
            'conditional_value': clean(value), 'reference_conditional_value': float(ref_value),
            'gain': clean(gain), 'reference_gain': clean(ref_gain),
            'gain_relative_error': float(np.linalg.norm(gain.numpy()[0]-ref_gain)/np.linalg.norm(ref_gain)),
            'conditional_covariance': clean(residual_covariance),
            'reference_conditional_covariance': clean(ref_residual_covariance),
            'conditional_covariance_absolute_error': float(np.linalg.norm(residual_covariance-ref_residual_covariance)),
            'conditional_rank': clean(conditional_rank), 'on_support': clean(on_support)})
    save(request, 'actual-svd-scale-attribution.json', {'role': 'explanatory_existing_endpoint_scale_audit',
        'records': records, 'nonclaims': ['A diagnostic execution is not a passing numerical qualification.',
            'No runtime/default/tolerance change; actual route discrepancies remain repair triggers.']})
    assert all(np.isfinite(row['gain_relative_error']) for row in records)
