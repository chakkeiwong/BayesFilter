"""Diagnostic attribution of the preserved partial-block strict-decision failure."""

import json
from dataclasses import replace
from decimal import Decimal, localcontext
from pathlib import Path

import tensorflow as tf

from bayesfilter.inference import sequential_map_covariance as public
from bayesfilter.inference.block_conditional_tf import ConditionalSequentialProgram
from bayesfilter.inference.block_coordinate_center import BlockCoordinateCenterBlock
from bayesfilter.inference.sequential_controller_report import sequential_result
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_block_coordinate_center import _sequential_config
from tests.test_filter_repair_block_controller import _diagnostic_json

D = tf.float64


def test_partial_block_rounding_attribution(request):
    precision = tf.constant([[4., .2, .1, 1.], [.2, 5., .5, .2],
        [.1, .5, 6., .3], [1., .2, .3, 5.]], D)
    mode = tf.constant([.2, -.1, .3, -.2], D)
    center, scale = tf.fill([4], tf.constant(.03, D)), tf.fill([4], tf.constant(1.1, D))
    calls = tf.Variable(0, dtype=tf.int64)

    def full(point):
        calls.assign_add(1)
        delta = point - mode
        score = -tf.linalg.matvec(precision, delta)
        return .5 * tf.reduce_sum(delta * score), score

    def conditional(point):
        value, score = full(tf.concat([center[:1], point, center[3:]], 0))
        return value, score[1:3]

    checkpoint = FrozenCheckpoint('3582b4ac', 'block_rounding')
    original = checkpoint.load('bayesfilter.inference.sequential_map_covariance')
    rows = []
    for movement in (False, True):
        cfg = replace(_sequential_config(), record_refinement_movement_diagnostics=movement)
        results = {}
        for name, module in (('original', original), ('public', public)):
            calls.assign(0)
            result = module.estimate_sequential_map_covariance(conditional, center[None, 1:3],
                scale=scale[1:3], config=cfg)
            results[name] = {'result': result.payload(), 'calls': int(calls)}
        calls.assign(0)
        owner = ConditionalSequentialProgram(full, None, 4, BlockCoordinateCenterBlock('wide', 1, 3, cfg))
        result = sequential_result(owner.compiled(center, scale), cfg, 1, 2)
        results['dynamic'] = {'result': result.payload(), 'calls': int(calls)}
        rows.append({'movement_instrumentation': movement, 'arms': results})
    root = Path(request.config.getoption('xmlpath')).parent
    with (root / 'partial-block-rounding.json').open('x') as out:
        json.dump(_diagnostic_json({'role': 'explanatory attribution, not numerical qualification',
            'records': rows, 'original_source_sha256': checkpoint.hashes()}), out, indent=2, allow_nan=False)
        out.write('\n')


def test_identical_point_target_arithmetic(request):
    # Exact failing predecessor and proposal, preserved in03029. No changed
    # initializer config or tolerance is used to hide the strict-decision gap.
    points = tf.constant([[.03, -.10154621852803877, .29146218484342384, .03],
        [.03, -.10154621848739497, .2914621848739496, .03]], D)
    precision_rows = [[4., .2, .1, 1.], [.2, 5., .5, .2], [.1, .5, 6., .3], [1., .2, .3, 5.]]
    mode_rows = [.2, -.1, .3, -.2]
    precision, mode = tf.constant(precision_rows, D), tf.constant(mode_rows, D)

    def components(point):
        delta = point - mode
        score = -tf.linalg.matvec(precision, delta)
        products = delta * score
        return {'delta': delta, 'score': score, 'products': products,
            'sum': tf.reduce_sum(products), 'value': .5 * tf.reduce_sum(products)}

    def value_only(point):
        delta = point - mode
        score = -tf.linalg.matvec(precision, delta)
        return .5 * tf.reduce_sum(delta * score), score

    signature = [tf.TensorSpec([4], D)]
    arms = {'eager': components,
        'graph': tf.function(components, input_signature=signature, jit_compile=False, autograph=False),
        'xla': tf.function(components, input_signature=signature, jit_compile=True, autograph=False)}
    plain = {name: tf.function(value_only, input_signature=signature, jit_compile=jit, autograph=False)
        for name, jit in (('graph', False), ('xla', True))}
    values = {}
    for name, function in arms.items():
        values[name] = [tf.nest.map_structure(lambda item: item.numpy().tolist(), function(point)) for point in points]
    plain_values = {name: [tf.nest.map_structure(lambda item: item.numpy().tolist(), function(point)) for point in points]
        for name, function in plain.items()}
    high_precision = []
    with localcontext() as context:
        context.prec = 100
        matrix = [[Decimal.from_float(item) for item in row] for row in precision_rows]
        exact_mode = [Decimal.from_float(item) for item in mode_rows]
        for point in points.numpy().tolist():
            delta = [Decimal.from_float(item) - center for item, center in zip(point, exact_mode, strict=True)]
            score = [-sum(coefficient * coordinate for coefficient, coordinate in zip(row, delta, strict=True)) for row in matrix]
            value = sum(left * right for left, right in zip(delta, score, strict=True)) / 2
            high_precision.append({'value': str(value), 'score': [str(item) for item in score]})
        improvement = str(Decimal(high_precision[1]['value']) - Decimal(high_precision[0]['value']))
    root = Path(request.config.getoption('xmlpath')).parent
    with (root / 'partial-block-target-arithmetic.json').open('x') as out:
        json.dump({'role': 'identical-point diagnostic; no policy waiver', 'points': points.numpy().tolist(),
            'arms': values, 'uninstrumented_arms': plain_values, 'decimal100': high_precision,
            'decimal100_improvement': improvement}, out, indent=2, allow_nan=False)
        out.write('\n')
    for name in ('graph', 'xla'):
        for actual, expected in zip(values[name], plain_values[name], strict=True):
            assert actual['value'] == expected[0]
            assert actual['score'] == expected[1]
