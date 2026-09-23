"""Original external TensorFlow cloud stream versus native XLA preparation."""

import ast
import gc
import hashlib
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.dense_initializer_random_tf import (
    STREAM_ID,
    make_dense_initializer_cloud_design,
)
from bayesfilter.ops.stateless_random_tf import (
    philox_normal_float64,
    philox_uniform_float64,
)
from tests.test_filter_repair_block_capture import stable_hlo
from tests.test_filter_repair_dense_initializer_cloud import SOURCE, SOURCE_SHA
from tests.test_filter_repair_geometry_control import clean, save

D = tf.float64


def original_cloud():
    source = SOURCE.read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == SOURCE_SHA
    tree = ast.parse(source)
    loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For)
        and isinstance(node.target, ast.Tuple) and ast.unparse(node.target) == '(partition_index, rows)')
    boundary = next(index for index, node in enumerate(loop.body)
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'positions')
    function = ast.parse('def draw(configuration, attempt, partition_index, rows, dimension):\n    pass\n').body[0]
    function.body = [*loop.body[:boundary], ast.parse('return directions, radii, offsets').body[0]]
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    excerpt = ast.unparse(module)
    namespace = {'tf': tf}
    exec(compile(module, str(SOURCE) + ':original_rng', 'exec'), namespace)  # noqa: S102
    return namespace['draw'], excerpt


@pytest.mark.parametrize('dimension', [1, 3, 23])
def test_original_dense_cloud_stream(dimension, request):
    rows = ([68, 68, 46, 46, 46] if dimension == 23 else
        [3 * dimension] * 2 + [2 * dimension] * 2 + [2 * dimension + 1])
    capacity = max(rows)
    generate = make_dense_initializer_cloud_design(dimension, 2, rows[0], rows[2], rows[-1], 2)
    draw, excerpt = original_cloud()

    @tf.function(input_signature=[tf.TensorSpec([2], tf.int32)], jit_compile=True, autograph=False)
    def random_words(seed):
        return (tf.random.stateless_uniform([capacity, 2], seed, dtype=tf.uint32,
            minval=None, maxval=None, alg='philox'),
            philox_uniform_float64([capacity, 1], seed),
            philox_normal_float64([capacity, dimension], seed))

    reports, hlos, outputs = [], [], []
    for seed, radius in (((215, 91), .13), ((317, -59), .21), ((215, 91), .13)):
        configuration = SimpleNamespace(seed=seed, curvature_radius=radius)
        radius_operand = tf.constant(radius, D)
        seed_operand = tf.constant(seed, tf.int32)
        with tf.GradientTape() as tape:
            tape.watch(radius_operand)
            cloud = generate(seed_operand, radius_operand)
            loss = tf.reduce_sum(cloud)
        derivative = tape.gradient(loss, radius_operand)
        expected = np.zeros(cloud.shape, np.float64)
        for attempt in range(2):
            for partition, count in enumerate(rows):
                _, _, offsets = draw(configuration, attempt, partition, count, dimension)
                expected[attempt, partition, :count] = offsets.numpy()
        actual = cloud.numpy()
        words, uniform, normal = random_words(seed_operand)
        expected_words = tf.random.stateless_uniform([capacity, 2], seed_operand, dtype=tf.uint32,
            minval=None, maxval=None, alg='philox')
        expected_uniform = tf.random.stateless_uniform([capacity, 1], seed_operand, dtype=D)
        expected_normal = tf.random.stateless_normal([capacity, dimension], seed_operand, dtype=D)
        np.testing.assert_array_equal(words, expected_words)
        np.testing.assert_array_equal(uniform, expected_uniform)
        np.testing.assert_allclose(normal, expected_normal, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
        assert np.isfinite(actual).all() and np.max(np.linalg.norm(actual, axis=-1)) <= radius * (1. + 1e-12)
        for partition, count in enumerate(rows):
            assert (actual[:, partition, count:] == 0.).all()
        assert derivative is None
        outputs.append(actual)
        reports.append({'seed': seed, 'radius': radius, 'actual': clean(actual), 'original': clean(expected),
            'maximum_cloud_error': float(np.max(np.abs(actual - expected))),
            'maximum_normal_error': float(np.max(np.abs(normal.numpy() - expected_normal.numpy()))),
            'words_exact': bool(np.array_equal(words, expected_words)),
            'uniform_exact': bool(np.array_equal(uniform, expected_uniform)), 'frozen_derivative': derivative is None})
        hlos.append(stable_hlo(generate.experimental_get_compiler_ir(seed_operand, radius_operand)(stage='hlo')))
    traces = generate.experimental_get_tracing_count()
    refs = {'program': weakref.ref(generate), 'graph': weakref.ref(generate.get_concrete_function().graph)}
    del generate
    gc.collect()
    released = {name: value() is None for name, value in refs.items()}
    report = {'dimension': dimension, 'rows': rows, 'stream': STREAM_ID, 'records': reports,
        'source': str(SOURCE), 'source_sha256': SOURCE_SHA, 'original_excerpt': excerpt,
        'excerpt_sha256': hashlib.sha256(excerpt.encode()).hexdigest(), 'trace_count': traces,
        'hlo_unchanged': len(set(hlos)) == 1, 'python_released': released,
        'nonclaims': ['Cloud preparation only; seeded full-controller and actual target qualification remain open.']}
    save(request, f'dense-rng-{dimension}.json', report)
    np.testing.assert_array_equal(outputs[0], outputs[2])
    assert not np.array_equal(outputs[0], outputs[1])
    assert traces == 1 and report['hlo_unchanged'] and all(released.values())
