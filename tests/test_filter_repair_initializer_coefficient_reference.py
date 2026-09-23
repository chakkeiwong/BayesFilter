"""Independent diagnostic of near-zero initializer curvature coefficients.

NumPy and Decimal serve only as references for identical stored design/response
arrays. No diagnostic correction, coefficient or threshold enters runtime.
"""

import inspect
import math
from decimal import Decimal

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.compiler.tf2xla.ops.gen_xla_ops import xla_svd

from bayesfilter.inference import quadratic_geometry as geometry
from bayesfilter.inference import quadratic_geometry_fit_tf as fitting
from bayesfilter.inference.quadratic_geometry_full_report import geometry_result
from bayesfilter.inference.quadratic_geometry_full_tf import (
    make_geometry_program,
    prepare_geometry_inputs,
)
from tests.test_filter_repair_geometry_control import clean, save, source
from tests.test_filter_repair_geometry_full import original
from tests.test_filter_repair_initializer_arithmetic import residual_candidate
from tests.test_filter_repair_initializer_native import fixture
from tests.test_filter_repair_initializer_rounding import _decimal_error, _decimal_lstsq

D = tf.float64


@pytest.mark.parametrize('batched', [False, True])
def test_identical_arrays_high_precision_coefficient_reference(batched, monkeypatch, request):
    callback, batch, _, cfg, _, _, _, scale, reset, log = fixture(3, 'iterative', batched)
    prepared = prepare_geometry_inputs(3, cfg)
    _, old = source('3582b4ac')
    old_design = old._score_curvature_design
    captured = []

    def capture(*args, **kwargs):
        design, response = old_design(*args, **kwargs)
        captured.append((design.copy(), response.copy()))
        return design, response

    # Preserve the untouched kernel as an observer-effect control.
    body = inspect.getsource(geometry._quadratic_fit_kernel)
    before = '        "loss": loss,'
    assert body.count(before) == 1
    body = body.replace(before, '        "debug_design": design,\n        "debug_response": response,\n' + before)
    scope = dict(geometry._quadratic_fit_kernel.__globals__)
    exec(compile(body, 'diagnostic_coefficient_observer', 'exec'), scope)  # noqa: S102
    plain = make_geometry_program(callback, 3, cfg, batched_callback=batch)
    with monkeypatch.context() as patch:
        patch.setattr(fitting, '_quadratic_fit_kernel', scope['_quadratic_fit_kernel'])
        observed = make_geometry_program(callback, 3, cfg, batched_callback=batch)
    with monkeypatch.context() as patch:
        patch.setattr(fitting, '_quadratic_fit_kernel', residual_candidate())
        corrected = make_geometry_program(callback, 3, cfg, batched_callback=batch)

    @tf.function(input_signature=[tf.TensorSpec([36, 2], D), tf.TensorSpec([36], D)],
        jit_compile=True, autograph=False)
    def arithmetic(design, response):
        q, r = tf.linalg.qr(design, full_matrices=False)
        small = xla_svd(r, max_iter=100, epsilon=math.ulp(1.), precision_config='')
        left_r = tf.ensure_shape(small.u, [2, 2])
        right = tf.ensure_shape(small.v, [2, 2])
        singular = tf.ensure_shape(small.s, [2])
        inverse = tf.where(singular > tf.reduce_max(singular) * (36 * math.ulp(1.)),
            tf.math.reciprocal(singular), tf.zeros_like(singular))
        left = q @ left_r

        def solve(rhs):
            return tf.linalg.matvec(right, inverse * tf.linalg.matvec(left, rhs, transpose_a=True))

        raw = solve(response)
        return {'raw': raw, 'corrected': raw + solve(response - tf.linalg.matvec(design, raw))}

    records = []
    # Both adjacent centers and the later centers are from preserved02775.
    for value in (.17794228634059522, .17794228634059525,
                  .2558845726811768, .33382685902173204):
        center = tf.fill([3], tf.constant(value, D))
        inputs = (center, scale, *prepared)
        reset()
        with monkeypatch.context() as patch:
            patch.setattr(old, '_score_curvature_design', capture)
            expected, hashes = original(callback, batch, cfg, inputs, monkeypatch)
        old_matrix, old_rhs = captured[-1]
        old_calls = log()
        reset()
        untouched = plain(*inputs)
        plain_calls = log()
        reset()
        measured = observed(*inputs)
        reset()
        candidate = corrected(*inputs)
        corrected_calls = log()
        fit = measured['fit_result']['fit']
        rows = int(measured['partition']['training_count']) * 3
        matrix = fit['debug_design'][:rows].numpy()
        rhs = fit['debug_response'][:rows].numpy()
        arms = {}
        for name, a, b in (('original_inputs', old_matrix, old_rhs), ('native_inputs', matrix, rhs)):
            assert a.shape == (36, 2) and b.shape == (36,)
            singular = np.linalg.svd(a, compute_uv=False)
            assert singular[-1] / singular[0] > .01
            high = _decimal_lstsq(a, b[:, None], 100)
            lower = _decimal_lstsq(a, b[:, None], 70)
            agreement = _decimal_error(high, lower)
            assert agreement < Decimal('1e-60')
            tf_values = arithmetic(tf.constant(a, D), tf.constant(b, D))
            values = {'numpy_lstsq': np.linalg.lstsq(a, b, rcond=None)[0],
                **{key: tensor.numpy() for key, tensor in tf_values.items()}}
            arms[name] = {'design': a, 'response': b, 'condition': singular[0] / singular[-1],
                'decimal_reference': [str(row[0]) for row in high],
                'decimal_agreement': str(agreement),
                'solutions': {key: {'value': x, 'signed_mu': float(x[1]),
                    'coefficient_error': [str(Decimal.from_float(float(item)) - reference[0])
                        for item, reference in zip(x, high, strict=True)]} for key, x in values.items()}}
        plain_fit = untouched['fit_result']['fit']
        records.append({'center': center, 'expected': expected,
            'plain': clean(geometry_result(untouched, center, scale, cfg, batched=batched).payload(include_arrays=True)),
            'corrected': clean(geometry_result(candidate, center, scale, cfg, batched=batched).payload(include_arrays=True)),
            'plain_calls': plain_calls, 'corrected_calls': corrected_calls, 'original_calls': old_calls,
            'identical_design': np.array_equal(old_matrix, matrix), 'identical_response': np.array_equal(old_rhs, rhs),
            'observer_preserves_coefficients': float(plain_fit['raw_lambda0']) == float(fit['raw_lambda0'])
                and np.array_equal(plain_fit['raw_mu'].numpy(), fit['raw_mu'].numpy()),
            'arms': arms, 'original_sources': hashes})
    save(request, f'initializer-coefficient-reference-{batched}.json', {
        'role': 'identical-array independent coefficient attribution; no runtime repair or clipping-count waiver',
        'records': records})
    assert len(records) == len(captured) == 4
    assert arithmetic.experimental_get_tracing_count() == 1
