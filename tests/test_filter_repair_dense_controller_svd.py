"""Diagnostic attribution of the small-matrix XLA stability-norm discrepancy."""

import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from tests.test_filter_repair_geometry_control import clean, save

D = tf.float64


def test_identical_matrix_svd_scaling(request):
    source = (Path(request.config.getoption('xmlpath')).parent.parent /
        'run-03469/dense-controller-3-healthy.json')
    captured = json.loads(source.read_text())

    @tf.function(input_signature=[tf.TensorSpec([3, 3], D)], jit_compile=True, autograph=False)
    def probe(matrix):
        magnitude = tf.reduce_max(tf.abs(matrix))
        safe = tf.where(magnitude > 0., magnitude, tf.constant(1., D))
        return (tf.linalg.svd(matrix, compute_uv=False),
            tf.linalg.svd(matrix / safe, compute_uv=False) * magnitude)

    records = []
    for index, row in enumerate(captured['records']):
        for arm in ('original', 'actual'):
            fits = row[arm]['attempts'][0]['curvature']['fits']
            matrices = [np.asarray(fit['precision_z']) for fit in fits if fit['family'] == 'factor_1']
            difference = .5 * ((matrices[0] + matrices[0].T) - (matrices[1] + matrices[1].T))
            raw, scaled = probe(tf.constant(difference, D))
            reference = np.linalg.svd(difference, compute_uv=False)
            records.append({'iteration': index, 'arm': arm, 'difference': clean(difference),
                'raw_singular_values': clean(raw), 'scaled_singular_values': clean(scaled),
                'numpy_reference': clean(reference), 'matrix_magnitude': float(np.max(np.abs(difference))),
                'raw_relative_error': float(np.max(np.abs(raw.numpy() - reference)) / reference[0]),
                'scaled_relative_error': float(np.max(np.abs(scaled.numpy() - reference)) / reference[0])})
    examples = []
    for multiplier in (0., 1e-140, 1e-12, 1., 1e140):
        matrix = np.asarray([[2., 1., -.5], [1., 3., .25], [-.5, .25, 1.]]) * multiplier
        raw, scaled = probe(tf.constant(matrix, D))
        reference = np.linalg.svd(matrix, compute_uv=False)
        examples.append({'multiplier': multiplier, 'raw': clean(raw), 'scaled': clean(scaled),
            'reference': clean(reference)})
    report = {'role': 'explanatory_same_matrix_attribution', 'source': str(source),
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'records': records,
        'scale_examples': examples, 'trace_count': probe.experimental_get_tracing_count(),
        'nonclaims': ['No runtime or comparator change; does not qualify the full controller.']}
    save(request, 'dense-controller-svd-attribution.json', report)
    for row in records:
        assert row['scaled_relative_error'] < 1e-12
    for row in examples:
        np.testing.assert_allclose(row['scaled'], row['reference'], rtol=1e-12, atol=0.)
    assert report['trace_count'] == 1
