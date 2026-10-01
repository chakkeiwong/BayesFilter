"""Independent reference checks of the algebraic clipping-tie runtime repair.

NumPy supplies independent covariance/eigensystem checks and saved references.
The pinned source transformation is diagnostic; production uses only the
shared runtime. This does not qualify optimization or downstream admission.
"""

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from bayesfilter.inference import factor_correlation_geometry as factor
from tests.test_filter_repair_fixed_fitting_localization import _baseline
from tests.test_filter_repair_geometry_control import clean, save

D = tf.float64
FIXTURE = Path(__file__).parent / 'fixtures/filter_repair_factor_clipped_anchor_04578.json'
FIXTURE_SHA = 'fdaa20c0d78b6e94878cccd9a7ed036c8acfa9bf02b87779b8e5a2e9e5b98e99'


def before_initialization_source():
    source = subprocess.check_output(['git', 'show',
        '5d398a45b:bayesfilter/inference/factor_correlation_geometry.py'],
        cwd=Path(__file__).resolve().parents[1], text=True)
    node = next(node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == '_initial_factor_state')
    return ast.get_source_segment(source, node)


def compile_initialization(source):
    namespace = dict(vars(factor))
    exec(compile(source, 'diagnostic_exact_clipped_norm_anchor', 'exec'), namespace)  # noqa: S102
    return namespace['_initial_factor_state']


def proposed_initialization():
    source = before_initialization_source()
    expression = 'tf.minimum(tf.reshape(row_norms, [-1]), tf.constant(0.8 * bound, tf.float64))'
    for previous in ('tf.abs(loadings[:, 0])', 'tf.linalg.norm(loadings, axis=1)'):
        old = f'tf.argmax({previous}, output_type=tf.int32)'
        assert source.count(old) == 1
        source = source.replace(old, f'tf.argmax({expression}, output_type=tf.int32)')
    return compile_initialization(source), source


def initialization_record(function, covariance, count, jit):
    deviations, loadings, anchors = function(covariance, factor_count=count,
        loading_margin=1e-6, jit_compile=jit)
    config = factor.FactorCorrelationGeometryConfig(factor_count=count)
    raw = factor._encode_state(deviations, loadings, anchors, config)
    decoded, _, _ = factor._decode_covariance(raw, dimension=covariance.shape[0],
        anchors=anchors, config=config)
    return {'deviations': deviations, 'loadings': loadings, 'anchors': tf.stack(anchors),
        'raw': raw, 'decoded_covariance': decoded,
        'gram': tf.matmul(loadings, loadings, transpose_b=True)}


def test_exact_clipped_norm_anchor_proposal(request):
    tf.config.experimental.enable_tensor_float_32_execution(False)
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA
    fixture = json.loads(FIXTURE.read_text())
    proposed, transformed = proposed_initialization()
    before_source = before_initialization_source()
    before_function = compile_initialization(before_source)
    original = _baseline()
    cases = list(fixture['cases'])
    for row in fixture['cases']:
        matrix = np.asarray(row['covariance'])
        cases.append({'name': row['name'] + '_reversed', 'covariance': matrix[::-1, ::-1]})
    cases.extend([
        {'name': 'separated', 'covariance': np.diag([.8, 1.1, 1.7]) + .02 * np.ones((3, 3))},
        {'name': 'identity', 'covariance': np.eye(3)},
        {'name': 'nearly_repeated', 'covariance': np.eye(3) + 1e-9 * np.diag([1., 2., 3.])},
    ])
    records, failures = [], []
    programs = {}
    for case in cases:
        covariance = tf.constant(case['covariance'], D)
        dimension = covariance.shape[0]
        correlation = np.asarray(covariance) / np.sqrt(np.diag(covariance))[:, None] / np.sqrt(np.diag(covariance))[None, :]
        eigenvalues, eigenvectors = np.linalg.eigh(correlation)
        for count in (1, 2):
            ref_std, ref_load, ref_anchors = original._initial_factor_state(covariance,
                factor_count=count, loading_margin=1e-6)
            independent_loadings = eigenvectors[:, -count:] * np.sqrt(np.maximum(eigenvalues[-count:] - 1., 1e-6))
            unclipped_norm = np.linalg.norm(independent_loadings, axis=1)
            cap = .8 * np.sqrt(1. - 1e-6)
            clipped_indices = np.flatnonzero(unclipped_norm > cap)
            row = {'case': case['name'], 'factor_count': count,
                'original': clean({'deviations': ref_std, 'loadings': ref_load, 'anchors': ref_anchors}),
                'independent_clipped_indices': clipped_indices.tolist(), 'variants': {}}
            for mode in ('graph', 'xla'):
                key = (dimension, count, mode)
                if key not in programs:
                    def evaluate(matrix, count=count, mode=mode):
                        return {'current': initialization_record(before_function, matrix, count, mode == 'xla'),
                            'runtime': initialization_record(factor._initial_factor_state, matrix, count, mode == 'xla'),
                            'proposed': initialization_record(proposed, matrix, count, mode == 'xla')}
                    programs[key] = tf.function(evaluate,
                        input_signature=[tf.TensorSpec([dimension, dimension], D)],
                        jit_compile=mode == 'xla', autograph=False)
                result = programs[key](covariance)
                before, after = result['current'], result['proposed']
                for name in after:
                    np.testing.assert_allclose(result['runtime'][name], after[name], atol=1e-10, rtol=1e-10)
                delta = float(tf.reduce_max(tf.abs(after['decoded_covariance'] - before['decoded_covariance'])))
                bound = 16 * dimension * sys.float_info.epsilon * max(1., float(tf.linalg.norm(before['decoded_covariance'])))
                row['variants'][mode] = {**clean(result), 'covariance_max_error': delta, 'covariance_roundoff_bound': bound}
                if delta > bound:
                    failures.append({'case': case['name'], 'count': count, 'mode': mode, 'failure': 'changed_initial_covariance'})
                if len(clipped_indices) and int(after['anchors'][0]) != int(clipped_indices[0]):
                    failures.append({'case': case['name'], 'count': count, 'mode': mode, 'failure': 'not_first_clipped_maximum'})
                if case['name'] == 'separated':
                    for name in before:
                        if not np.allclose(after[name], before[name], atol=1e-10, rtol=1e-10):
                            failures.append({'case': case['name'], 'count': count, 'mode': mode, 'failure': name})
            records.append(row)
    report = {'schema': 'filter_clipped_anchor_algebraic_trial.v1', 'records': records, 'failures': failures,
        'fixture_sha256': hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
        'transformed_source': transformed, 'transformed_source_sha256': hashlib.sha256(transformed.encode()).hexdigest(),
        'before_source': before_source, 'before_commit': '5d398a45b',
        'before_source_sha256': hashlib.sha256(before_source.encode()).hexdigest(),
        'trace_counts': {str(key): value.experimental_get_tracing_count() for key, value in programs.items()},
        'runtime_changed': True, 'optimizer_or_admission_qualified': False}
    save(request, 'factor-clipped-anchor-trial.json', report)
    assert not failures, failures
    assert all(count == 1 for count in report['trace_counts'].values())
