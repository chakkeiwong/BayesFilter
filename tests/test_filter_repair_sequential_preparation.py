"""Independent and pinned authorities for native sequential preparation.

The pinned helpers execute afresh as engineering references. No historical
research result or HMC/NeuTra admission is inferred from these checks.
"""

import importlib.util
import subprocess
import sys

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import sequential_map_covariance as candidate
from bayesfilter.inference import sequential_preparation_tf as native

D = tf.float64


@pytest.fixture(scope="module")
def baseline():
    source = subprocess.check_output(["git", "show",
        "3582b4ac5fea67fb5da7fa60a1cbfaf19df35adf:bayesfilter/inference/sequential_map_covariance.py"], text=True)
    spec = importlib.util.spec_from_loader("sequential_preparation_pinned_reference", loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(source, spec.name, "exec"), module.__dict__)  # noqa: S102 - pinned diagnostic reference
    return module


@pytest.mark.parametrize("count,dimension,radius", [(1, 1, .25), (12, 2, .3), (21, 3, .25), (9, 5, 1.)])
@pytest.mark.parametrize("orthogonal", [False, True])
def test_clouds_preserve_original_seed_frame_and_pair_order(baseline, count, dimension, radius, orthogonal):
    function = "_orthogonal_antithetic_cloud" if orthogonal else "_antithetic_cloud"
    for seed in ((2026, 717), (19, 71237)):
        expected = getattr(baseline, function)(count, dimension, radius, seed)
        actual = getattr(candidate, function)(count, dimension, radius, seed)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
        assert actual.shape == (count, dimension)
        if orthogonal:
            np.testing.assert_array_equal(np.asarray(actual)[1::2], -np.asarray(actual)[0:count - 1:2])
        else:
            half = (count + 1) // 2
            np.testing.assert_array_equal(np.asarray(actual)[half:], -np.asarray(actual)[:count - half])
    program = native.cloud_program(count, dimension, orthogonal)
    assert program.experimental_get_tracing_count() == 1
    assert "HloModule" in program.experimental_get_compiler_ir(
        tf.constant(radius, D), tf.constant(seed, tf.int32))(stage="hlo")


@pytest.mark.parametrize("count", [3, 7])
@pytest.mark.parametrize("batched", [False, True])
def test_complete_cloud_values_scores_and_enclosing_derivative(baseline, count, batched):
    precision = tf.constant([[2., .3], [.3, 4.]], D)

    def scalar(point):
        score = -tf.linalg.matvec(precision, point)
        return .5 * tf.reduce_sum(point * score), score

    def batch(points):
        scores = -tf.einsum("ij,bj->bi", precision, points)
        return .5 * tf.reduce_sum(points * scores, axis=1), scores

    points = tf.reshape(tf.linspace(tf.constant(-1., D), tf.constant(1., D), 2 * count), [count, 2])
    batch_target = batch if batched else None
    options = {"dimension": 2, "batched_value_and_score_fn": batch_target}
    expected = baseline._evaluate_cloud(scalar, points, **options)
    actual = candidate._evaluate_cloud(scalar, points, **options)
    np.testing.assert_allclose(actual[0], expected[0], atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(actual[1], expected[1], atol=1e-10, rtol=1e-10)
    program = native.evaluation_program(scalar, batch_target, count, 2)
    program(points * 2.)
    assert program.experimental_get_tracing_count() == 1
    graph = program.get_concrete_function().graph.as_graph_def()
    nodes = list(graph.node) + [node for fn in graph.library.function for node in fn.node_def]
    assert not any(node.op in ("PyFunc", "EagerPyFunc", "PyFuncStateless") for node in nodes)
    assert "HloModule" in program.experimental_get_compiler_ir(points)(stage="hlo")

    @tf.function(input_signature=[tf.TensorSpec([count, 2], D)], jit_compile=True, autograph=False)
    def derivative(rows):
        with tf.GradientTape() as tape:
            tape.watch(rows)
            values, _ = candidate._evaluate_cloud(scalar, rows, **options)
            total = tf.reduce_sum(values)
        return tape.gradient(total, rows)

    np.testing.assert_allclose(derivative(points), expected[1], atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("precision,linear,radius", [
    ([[2., 0.], [0., 4.]], [2., 0.], 1.),
    ([[2., 0.], [0., 4.]], [2., 0.], np.nextafter(1., 0.)),
    ([[2., .3], [.3, 4.]], [2., -3.], .2),
    ([[2., .3], [.3, 4.]], [2., -3.], 10.),
    ([[1e-4, 1e-3], [1e-3, 1e4]], [1e-3, 2.], .25),
    ([[2., .3], [.3, 4.]], [0., 0.], .25),
    ([[3., .2, -.1], [.2, 2., .4], [-.1, .4, 4.]], [1., -2., 3.], .1),
])
def test_trust_region_preserves_step_boundary_and_improvement(baseline, precision, linear, radius):
    matrix, vector = tf.constant(precision, D), tf.constant(linear, D)
    expected = baseline._solve_trust_region_tf(matrix, vector, radius)
    actual = candidate._solve_trust_region_tf(matrix, vector, radius)
    assert actual["boundary_active"] == expected["boundary_active"]
    np.testing.assert_allclose(actual["step"], expected["step"], atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(actual["predicted_improvement"], expected["predicted_improvement"], atol=1e-10, rtol=1e-10)
    step = np.asarray(actual["step"])
    gradient = np.asarray(linear) - np.asarray(precision) @ step
    if actual["boundary_active"]:
        assert np.linalg.norm(step) == pytest.approx(radius, abs=1e-10)
        multiplier = np.dot(gradient, step) / np.dot(step, step)
        assert multiplier >= -1e-10
        np.testing.assert_allclose(gradient, multiplier * step, atol=1e-10, rtol=1e-10)
    else:
        assert np.linalg.norm(step) <= radius + 1e-10
        np.testing.assert_allclose(gradient, 0., atol=1e-10, rtol=1e-10)
    program = native.trust_region_program(len(linear))
    assert program.experimental_get_tracing_count() == 1
    assert "HloModule" in program.experimental_get_compiler_ir(matrix, vector, tf.constant(radius, D))(stage="hlo")


def test_native_frame_graph_growth_is_bounded():
    counts = []
    for count in (8, 80):
        graph = native.cloud_program(count, 2, True).get_concrete_function().graph.as_graph_def()
        nodes = list(graph.node) + [node for fn in graph.library.function for node in fn.node_def]
        counts.append(len(nodes))
        assert any(node.op in ("While", "StatelessWhile") for node in nodes)
    assert abs(counts[0] - counts[1]) <= 2
