"""Diagnostic-only attribution of initializer least-squares rounding.

NumPy is the independent pinned-reference backend in this test. No diagnostic
coefficient or altered solver is installed in an admitted numerical path.
"""

import inspect
import math

import numpy as np
import tensorflow as tf
from tensorflow.compiler.tf2xla.ops.gen_xla_ops import xla_svd

from bayesfilter.inference import quadratic_geometry as geometry
from bayesfilter.inference import quadratic_geometry_fit_tf as fitting
from bayesfilter.inference.quadratic_geometry_full_tf import (
    make_geometry_program,
    prepare_geometry_inputs,
)
from bayesfilter.inference.quadratic_initializer_report import initializer_result
from bayesfilter.inference.quadratic_initializer_tf import (
    make_quadratic_initializer_program,
)
from tests.test_filter_repair_geometry_control import clean, save, source
from tests.test_filter_repair_geometry_full import original
from tests.test_filter_repair_initializer_native import (
    comparison,
    fixture,
)
from tests.test_filter_repair_initializer_native import (
    original as original_initializer,
)
from tests.test_filter_repair_quadratic_batches import _equal_records

D = tf.float64


def test_initializer_least_squares_attribution(monkeypatch, request):
    callback, _, _, cfg, _, _, _, scale, reset, _ = fixture(1, 'iterative', False)
    prepared = prepare_geometry_inputs(1, cfg)
    _, original_module = source('3582b4ac')
    original_design = original_module._score_curvature_design
    captured = []

    def capture(*args, **kwargs):
        design, response = original_design(*args, **kwargs)
        captured.append((design.copy(), response.copy()))
        return design, response

    # Extra graph outputs can affect fusion. Keep the untouched result as an
    # explicit control and report any observer effect instead of hiding it.
    body = inspect.getsource(geometry._quadratic_fit_kernel)
    before = '        "loss": loss,'
    assert body.count(before) == 1
    body = body.replace(before, '''        "debug_design": design,
        "debug_response": response,
        "debug_q": reduced_q,
        "debug_r": reduced_r,
        "debug_raw": raw,
        "loss": loss,''')
    scope = dict(geometry._quadratic_fit_kernel.__globals__)
    exec(compile(body, 'diagnostic_geometry_fit_observer', 'exec'), scope)  # noqa: S102
    plain = make_geometry_program(callback, 1, cfg)
    with monkeypatch.context() as patch:
        patch.setattr(fitting, '_quadratic_fit_kernel', scope['_quadratic_fit_kernel'])
        observed = make_geometry_program(callback, 1, cfg)

    @tf.function(input_signature=[tf.TensorSpec([12, 1], D), tf.TensorSpec([12], D)],
                 autograph=False, jit_compile=True)
    def arithmetic(design, response):
        q, r = tf.linalg.qr(design, full_matrices=False)
        small = xla_svd(r, max_iter=100, epsilon=math.ulp(1.), precision_config='')
        left, right = tf.ensure_shape(small.u, [1, 1]), tf.ensure_shape(small.v, [1, 1])
        inverse = tf.math.reciprocal(tf.ensure_shape(small.s, [1]))
        u = tf.matmul(q, left)

        def solve(rhs):
            return tf.linalg.matvec(right, inverse * tf.linalg.matvec(u, rhs, transpose_a=True))

        raw = solve(response)
        # Same retained QR/SVD factors, with a diagnostic residual correction.
        # This is not a normal-equation substitute or a rank-policy change.
        residual = response - tf.linalg.matvec(design, raw)
        refined = raw + solve(residual)
        whole = xla_svd(design, max_iter=100, epsilon=math.ulp(1.), precision_config='')
        whole_u = tf.ensure_shape(whole.u, [12, 12])[:, :1]
        whole_v = tf.ensure_shape(whole.v, [1, 1])
        direct = tf.linalg.matvec(whole_v, tf.linalg.matvec(whole_u, response, transpose_a=True)
                                 / tf.ensure_shape(whole.s, [1]))
        return {'q': q, 'r': r, 'raw': raw, 'residual': residual,
                'refined': refined, 'direct_svd': direct}

    records = []
    for center in (.1, .23499999999995466, .2349999999999547, .34, .3400000000000001):
        inputs = (tf.constant([center], D), scale, *prepared)
        reset()
        with monkeypatch.context() as patch:
            patch.setattr(original_module, '_score_curvature_design', capture)
            expected, hashes = original(callback, None, cfg, inputs, monkeypatch)
        old_design, old_response = captured[-1]
        reset()
        untouched = plain(*inputs)
        reset()
        measured = observed(*inputs)
        fit = measured['fit_result']['fit']
        count = int(measured['partition']['training_count'])
        design, response = fit['debug_design'][:count], fit['debug_response'][:count]
        arms = {}
        for name, matrix, rhs in (('original_inputs', old_design, old_response),
                                  ('native_inputs', design.numpy(), response.numpy())):
            q, r = np.linalg.qr(matrix, mode='reduced')
            u, s, vh = np.linalg.svd(r, full_matrices=False)
            arms[name] = {'design': matrix, 'response': rhs,
                'numpy_lstsq': np.linalg.lstsq(matrix, rhs, rcond=None)[0],
                'numpy_qr_svd': vh.T @ ((q @ u).T @ rhs / s),
                'tensorflow': arithmetic(tf.constant(matrix, D), tf.constant(rhs, D))}
        records.append({'center': center, 'expected': expected,
            'untouched': untouched, 'observed': measured, 'arms': arms,
            'identical_design': np.array_equal(old_design, design.numpy()),
            'identical_response': np.array_equal(old_response, response.numpy()),
            'observer_preserves_coefficient': float(untouched['fit_result']['fit']['raw_lambda0']) == float(fit['raw_lambda0']),
            'original_sources': hashes})
    save(request, 'initializer-least-squares-attribution.json', clean({
        'role': 'explanatory arithmetic attribution only; no numerical or status waiver',
        'records': records}))
    assert len(captured) == len(records) == 5
    assert arithmetic.experimental_get_tracing_count() == 1


def residual_candidate():
    """A single residual correction with the same rank-selected QR/SVD factors."""
    body = inspect.getsource(geometry._quadratic_fit_kernel)
    before = '    lambda0 = tf.maximum(floor, raw[0])'
    assert body.count(before) == 1
    body = body.replace(before, '''    residual = response - tf.linalg.matvec(design, raw)
    if active_rows is None:
        correction = tf.linalg.matvec(right, inverse * tf.linalg.matvec(left, residual, transpose_a=True))
    else:
        correction = compact_solution(reduced_q, left_r[:, :extent], right, inverse,
            residual, active_rows, dim, minimum_active_rows)
    raw = raw + correction
    lambda0 = tf.maximum(floor, raw[0])''')
    scope = dict(geometry._quadratic_fit_kernel.__globals__)
    exec(compile(body, 'diagnostic_geometry_residual_correction', 'exec'), scope)  # noqa: S102
    return scope['_quadratic_fit_kernel']


def test_initializer_residual_correction_candidate(monkeypatch, request):
    """Complete original records decide whether the candidate merits expansion."""
    records = []
    for dimension in (1, 3):
        for batched in (False, True):
            callback, batch, locator, cfg, mass, iterative, initial, scale, reset, log = fixture(
                dimension, 'iterative', batched)
            inputs = (initial, scale, *prepare_geometry_inputs(dimension, cfg))
            configs = locator, cfg, mass, iterative
            expected, events_expected, hashes, compatibility = original_initializer(
                callback, batch, configs, inputs, monkeypatch)
            calls_expected = log()
            reset()
            with monkeypatch.context() as patch:
                patch.setattr(fitting, '_quadratic_fit_kernel', residual_candidate())
                program = make_quadratic_initializer_program(callback, dimension, locator,
                    cfg, mass, iterative_config=iterative, batched_callback=batch)
                raw = program(*inputs)
            events = []
            actual = initializer_result(raw, initial, scale, locator, cfg, mass,
                iterative_config=iterative, batched=batched,
                fit_start_callback=lambda i, center, events=events: events.append(('start', i, clean(center))),
                iteration_callback=lambda row, events=events: events.append(('iteration', clean(row))))
            actual_payload, actual_calls = clean(actual.payload(include_arrays=True)), log()
            differences = {}
            for label, value, reference in (
                    ('record', comparison(actual_payload), comparison(expected)),
                    ('events', comparison(clean(events)), comparison(events_expected)),
                    ('calls', actual_calls, calls_expected)):
                try:
                    _equal_records(value, reference)
                except AssertionError as error:
                    differences[label] = str(error)
            records.append({'dimension': dimension, 'batched': batched,
                'actual': actual_payload, 'expected': expected, 'actual_calls': actual_calls,
                'expected_calls': calls_expected, 'actual_events': clean(events),
                'expected_events': events_expected, 'differences': differences,
                'original_sources': hashes, 'gpu_comparator': compatibility})
    save(request, 'initializer-residual-candidate.json', {
        'role': 'dedicated non-harm candidate evaluation; not installed runtime', 'records': records})
    assert all(not row['differences'] for row in records), [
        (row['dimension'], row['batched'], row['differences']) for row in records if row['differences']]
