"""Independent symmetry projection diagnostic before shared runtime adoption."""

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from tests.test_filter_repair_geometry_control import clean, save

RAW = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')
D = tf.float64


def test_final_inverse_projection_preserves_accuracy(request):
    tf.config.experimental.enable_tensor_float_32_execution(False)
    cases = []
    for number in (4606, 4607, 4608, 4609):
        report = json.loads((RAW / f'run-{number:05d}/dz5-exact-saved-fit.json').read_text())
        fits = report['raw']['fit']['fits']
        for family in (1, 2):
            for replicate in (0, 1):
                if fits['flags'][family][replicate][1]:
                    cases.append((f'{number}_factor{family}_replicate{replicate}',
                        np.asarray(fits['covariance'][family][replicate], dtype=float)))
    orthogonal = np.linalg.qr(np.arange(1., 26.).reshape(5, 5) + np.eye(5))[0]
    for scale in (1e-8, 1., 1e8):
        cases.append((f'analytic_scaled_{scale}', scale * (orthogonal * [1., 1., 2., 10., 1e4]) @ orthogonal.T))
    records, programs = [], {}
    for name, covariance in cases:
        dimension = len(covariance)
        authority = np.linalg.inv((covariance + covariance.T) / 2)
        for jit in (False, True):
            key = (dimension, jit)
            if key not in programs:
                def inverse(matrix):
                    raw = tf.linalg.cholesky_solve(tf.linalg.cholesky(matrix), tf.eye(matrix.shape[0], dtype=D))
                    return raw, .5 * (raw + tf.transpose(raw))
                programs[key] = tf.function(inverse,
                    input_signature=[tf.TensorSpec([dimension, dimension], D)], jit_compile=jit, autograph=False)
            raw, symmetric = programs[key](tf.constant(covariance, D))
            raw, symmetric = raw.numpy(), symmetric.numpy()
            norm = np.linalg.norm(authority, ord=2)
            cond = np.linalg.cond(covariance)
            residual = np.linalg.norm(covariance @ symmetric - np.eye(dimension), ord=2)
            backward = residual / (np.linalg.norm(covariance, ord=2) * np.linalg.norm(symmetric, ord=2) + 1.)
            row = {'case': name, 'jit_compile': jit, 'condition_number': cond,
                'covariance': covariance.tolist(), 'raw': raw.tolist(), 'symmetric': symmetric.tolist(),
                'relative_correction': float(np.linalg.norm(symmetric - raw, ord=2) / norm),
                'relative_inverse_error': float(np.linalg.norm(symmetric - authority, ord=2) / norm),
                'backward_error': float(backward), 'residual': float(residual)}
            records.append(row)
            np.testing.assert_array_equal(symmetric, symmetric.T)
            assert row['relative_correction'] <= 16 * dimension * sys.float_info.epsilon
            assert row['relative_inverse_error'] <= 16 * dimension * sys.float_info.epsilon * cond
            assert backward <= 16 * dimension * sys.float_info.epsilon
    # Derivative authority for trace(C^-1) is -(C^-1)^2 for symmetric C.
    matrix = tf.constant([[2., .2], [.2, 1.]], D)
    @tf.function(input_signature=[tf.TensorSpec([2, 2], D)], jit_compile=True, autograph=False)
    def derivative(matrix):
        with tf.GradientTape() as tape:
            tape.watch(matrix)
            sym = .5 * (matrix + tf.transpose(matrix))
            raw = tf.linalg.cholesky_solve(tf.linalg.cholesky(sym), tf.eye(2, dtype=D))
            loss = tf.linalg.trace(.5 * (raw + tf.transpose(raw)))
        return tape.gradient(loss, matrix)
    actual_derivative = derivative(matrix)
    independent = np.linalg.inv(matrix.numpy())
    np.testing.assert_allclose(actual_derivative.numpy(), -independent @ independent, atol=1e-12, rtol=1e-12)
    @tf.function(input_signature=[tf.TensorSpec([2, 2], D)], jit_compile=True, autograph=False)
    def invalid_inverse(matrix):
        raw = tf.linalg.cholesky_solve(tf.linalg.cholesky(matrix), tf.eye(2, dtype=D))
        return .5 * (raw + tf.transpose(raw))
    invalid = invalid_inverse(tf.constant([[1., 2.], [2., 1.]], D))
    assert not bool(tf.reduce_all(tf.math.is_finite(invalid)))
    save(request, 'factor-precision-symmetry-trial.json', {'schema': 'filter_factor_precision_symmetry_trial.v1',
        'runtime_changed': False, 'records': records, 'derivative': clean(actual_derivative),
        'invalid_remains_nonfinite': True,
        'trace_counts': {str(key): p.experimental_get_tracing_count() for key, p in programs.items()},
        'nonclaims': ['Small raw inverse skew is diagnosed, not a tolerance waiver or full fitter/consumer qualification.']})
