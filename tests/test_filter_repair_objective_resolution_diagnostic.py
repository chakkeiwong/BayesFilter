"""Uninstalled diagnostic candidate for exact-incumbent resolution reporting."""

import json
from pathlib import Path

import tensorflow as tf

from bayesfilter.inference.block_conditional_tf import ConditionalSequentialProgram
from bayesfilter.inference.block_coordinate_center import BlockCoordinateCenterBlock
from tests.test_block_coordinate_center import _sequential_config

D = tf.float64


@tf.function(input_signature=[tf.TensorSpec([None], D), tf.TensorSpec([None], D),
    tf.TensorSpec([None], tf.bool), tf.TensorSpec([None], tf.bool)],
    jit_compile=True, autograph=False)
def diagnostic_resolution_flags(before, after, promoted, executed):
    finite = tf.math.is_finite(before) & tf.math.is_finite(after)
    adjacent = tf.math.nextafter(before, tf.fill(tf.shape(before), tf.constant(float('inf'), D)))
    return executed & promoted & finite & (after > before) & (after <= adjacent)


def test_uninstalled_resolution_candidate_on_complete_records(request):
    cases = (
        ('preserved_partial', [[4., .2, .1, 1.], [.2, 5., .5, .2], [.1, .5, 6., .3], [1., .2, .3, 5.]],
            [.2, -.1, .3, -.2], 1, 3, .03, 1.1, True),
        ('partial_initial', [[4., .2, .1, 1.], [.2, 5., .5, .2], [.1, .5, 6., .3], [1., .2, .3, 5.]],
            [.2, -.1, .3, -.2], 1, 3, 0., 1., False),
        ('coupled_initial', [[4., 3.], [3., 4.]], [1., -1.], 0, 1, 0., 1., False),
        ('coupled_changed', [[4., 3.], [3., 4.]], [1., -1.], 0, 1, .03, 1.1, False),
    )
    rows = []
    for name, matrix, mode, start, stop, offset, multiplier, expected in cases:
        precision, mode = tf.constant(matrix, D), tf.constant(mode, D)

        def target(point, precision=precision, mode=mode):
            delta = point - mode
            score = -tf.linalg.matvec(precision, delta)
            return .5 * tf.reduce_sum(delta * score), score

        dimension = len(matrix)
        owner = ConditionalSequentialProgram(target, None, dimension,
            BlockCoordinateCenterBlock(name, start, stop, _sequential_config()))
        raw = owner.compiled(tf.fill([dimension], tf.constant(offset, D)), tf.fill([dimension], tf.constant(multiplier, D)))
        history = raw['lifecycle']['history']
        refine = history['refine']
        flags = diagnostic_resolution_flags(refine['selected_value'], refine['center_value'],
            refine['attempts']['promoted_without_acceptance'],
            (history['event'] == 3) & (refine['action'] >= 2))
        rows.append({'case': name, 'expected': expected, 'flags': flags.numpy().tolist(),
            'flagged': bool(tf.reduce_any(flags)), 'evaluations': int(raw['lifecycle']['evaluations']),
            'unchanged_runtime_status': int(raw['lifecycle']['status'])})
    root = Path(request.config.getoption('xmlpath')).parent
    with (root / 'objective-resolution-candidate.json').open('x') as out:
        json.dump({'role': 'uninstalled explanatory candidate; no runtime or comparison change', 'cases': rows}, out, indent=2)
        out.write('\n')
    for row in rows:
        assert row['flagged'] == row['expected'], row
