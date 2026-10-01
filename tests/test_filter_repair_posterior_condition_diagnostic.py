"""Diagnostic only: rank-two rejected-design rounding and independent references."""

import json
from pathlib import Path

import mpmath as mp
import numpy as np
import tensorflow as tf

from bayesfilter.ops.qr_lstsq_tf import complete_orthogonal_lstsq


def test_rejected_design_same_input_and_original_sensitivity(request):
    root = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')
    frozen = root / 'run-02376/posterior-cod-rank-ill_conditioned.json'
    record = json.loads(frozen.read_text())
    matrix, response = np.array(record['offsets']), np.array(record['response'])
    references = []
    reference_digits = []
    for precision in (100, 160):
        with mp.workdps(precision):
            a, b = mp.matrix(matrix.tolist()), mp.matrix(response.tolist())
            u, singular, v = mp.svd_r(a, full_matrices=False)
            # The same unchanged epsilon*3 threshold selects two directions.
            threshold = singular[0] * mp.mpf(float(np.finfo(float).eps)) * 3
            assert sum(singular[i] > threshold for i in range(3)) == 2
            solution = mp.zeros(3, 3)
            for k in range(2):
                for row in range(3):
                    for column in range(3):
                        solution[row, column] += v[k, row] * sum(u[i, k] * b[i, column] for i in range(33)) / singular[k]
            sym = (solution + solution.T) / 2
            references.append(np.array(sym.tolist(), dtype=float))
            reference_digits.append([[mp.nstr(sym[i, j], 90) for j in range(3)] for i in range(3)])
    np.testing.assert_array_equal(references[0], references[1])
    signature = [tf.TensorSpec([33, 3], tf.float64), tf.TensorSpec([33, 3], tf.float64)]

    @tf.function(input_signature=signature, autograph=False, jit_compile=False)
    def original(a, b):
        return tf.linalg.lstsq(a, b, fast=False)

    graph = tf.function(complete_orthogonal_lstsq, input_signature=signature, autograph=False, jit_compile=False)
    xla = tf.function(complete_orthogonal_lstsq, input_signature=signature, autograph=False, jit_compile=True)
    expected_solution = original(matrix, response).numpy()
    expected = .5 * (expected_solution + expected_solution.T)
    np.testing.assert_allclose(expected, record['original_precision'], atol=1e-14, rtol=1e-14)
    variants = {'original': (original, matrix, response), 'graph': (graph, matrix, response),
        'xla': (xla, matrix, response),
        'original_rhs_plus_ulp': (original, matrix, np.nextafter(response, np.inf)),
        'original_rhs_minus_ulp': (original, matrix, np.nextafter(response, -np.inf)),
        'original_design_plus_ulp': (original, np.nextafter(matrix, np.inf), response),
        'original_design_minus_ulp': (original, np.nextafter(matrix, -np.inf), response)}
    outcomes = {}
    for name, (program, a, b) in variants.items():
        solution = program(a, b).numpy()
        precision = .5 * (solution + solution.T)
        difference = np.abs(precision - expected)
        outcomes[name] = {'precision': precision.tolist(), 'solution': solution.tolist(),
            'failed_precision_entries_at_1e10': int(np.count_nonzero(difference > 1e-10 + 1e-10 * np.abs(expected))),
            'max_precision_difference': float(np.max(difference)),
            'matrix_relative_error_vs_original': float(np.linalg.norm(precision - expected) / np.linalg.norm(expected)),
            'matrix_relative_error_vs_mp': float(np.linalg.norm(precision - references[0]) / np.linalg.norm(references[0])),
            'relative_response_residual': float(np.linalg.norm(a @ solution - b) / np.linalg.norm(b))}
    report = {'role': 'explanatory_only_no_tolerance_waiver', 'frozen_inputs': str(frozen),
        'independent_reference': '100/160-digit SVD with unchanged two-direction truncation',
        'independent_reference_digits': reference_digits, 'results': outcomes,
        'scope': 'rejected rank-two ill-conditioned design; no accepted geometry or score change'}
    directory = Path(request.config.getoption('xmlpath')).parent
    with (directory / 'posterior-rejected-condition-diagnostic.json').open('x') as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write('\n')
