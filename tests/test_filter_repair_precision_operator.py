"""Independent operator norms and original stability decisions at small scales."""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from bayesfilter.inference import fixed_center_curvature as current
from tests.test_filter_repair_dense_validated_fit import original_fitter
from tests.test_filter_repair_fixed_stability import _compare, _fits, _thresholds
from tests.test_filter_repair_geometry_control import clean, save

D = tf.float64


def test_operator_norm_scale_and_zero(request):
    @tf.function(input_signature=[tf.TensorSpec([3, 3], D), tf.TensorSpec([], D)],
        jit_compile=True, autograph=False)
    def calculate(matrix, scale):
        return current._trace_normalized_operator(matrix, scale)

    records = []
    base = np.array([[2., 1., -.5], [1., 3., .25], [-.5, .25, 1.]])
    for multiplier in (0., 1e-140, 1e-12, 1., 1e140):
        matrix = multiplier * base
        scale = max(multiplier, 1e-15)
        expected = float(np.linalg.svd(matrix, compute_uv=False)[0] / scale)
        actual = float(calculate(tf.constant(matrix, D), tf.constant(scale, D)))
        records.append({'multiplier': multiplier, 'scale': scale, 'actual': actual, 'reference': expected})
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.)
    nearly_tied = tf.constant([[1., 1e-7, 0.], [1e-7, 1., 0.], [0., 0., .5]], D)
    tied_value = float(calculate(nearly_tied, tf.constant(1., D)))
    np.testing.assert_allclose(tied_value, 1. + 1e-7, rtol=1e-14, atol=0.)
    records.append({'case': 'analytic_nearly_tied', 'actual': tied_value, 'reference': 1. + 1e-7})
    save(request, 'precision-operator-scales.json', {'records': records,
        'trace_count': calculate.experimental_get_tracing_count()})
    assert calculate.experimental_get_tracing_count() == 1


def test_small_difference_preserves_original_records_and_cap_decisions(request):
    source = (Path(request.config.getoption('xmlpath')).parent.parent /
        'run-03469/dense-controller-3-healthy.json')
    captured = json.loads(source.read_text())
    original, hashes = original_fitter()
    reports = []
    for index, row in enumerate(captured['records'][:2]):
        for arm in ('original', 'actual'):
            fits = row[arm]['attempts'][0]['curvature']['fits']
            matrices = [np.asarray(fit['precision_z']) for fit in fits if fit['family'] == 'factor_1']
            before = original.compare_precision_geometry(*matrices, subspace_rank=1)
            after = current.compare_precision_geometry(*matrices, subspace_rank=1)
            left, right = map(np.asarray, matrices)
            difference = .5 * ((left + left.T) - (right + right.T))
            reference = np.linalg.svd(difference, compute_uv=False)[0] / max(abs(np.trace(left)), abs(np.trace(right)))
            np.testing.assert_allclose(after['trace_normalized_operator'], reference, rtol=1e-10, atol=0.)
            _compare(after, before)
            decisions = []
            for ratio in (.999, 1.001):
                cap = float(reference * ratio)
                old = original._family_stability(_fits(matrices),
                    _thresholds(original, trace_normalized_operator_cap=cap))
                new = current._family_stability(_fits(matrices),
                    _thresholds(current, trace_normalized_operator_cap=cap))
                _compare(new, old)
                assert new['passed'] is (ratio > 1.)
                decisions.append({'cap': cap, 'passed': new['passed']})
            reports.append({'iteration': index, 'arm': arm, 'original': before, 'current': after,
                'independent_operator': float(reference), 'decisions': decisions})
    save(request, 'precision-operator-original-comparisons.json', clean({'records': reports,
        'original_sources': hashes, 'fixture': str(source)}))
