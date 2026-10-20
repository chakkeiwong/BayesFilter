"""Original selection semantics, enclosing compilation and consumer checks."""

import importlib.util
import subprocess
import sys

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import sequential_map_covariance as candidate
from bayesfilter.inference import sequential_selection_tf as native
from tests.test_filter_repair_fixed_stability import _compare

D = tf.float64


@pytest.fixture(scope="module")
def previous():
    source = subprocess.check_output(["git", "show",
        "8f334b96:bayesfilter/inference/sequential_map_covariance.py"], text=True)
    spec = importlib.util.spec_from_loader("sequential_selection_pre_enclosure_reference", loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(source, spec.name, "exec"), module.__dict__)  # noqa: S102
    return module


def _reference(positions, values, scores, *, keep_first=False):
    rows = [(float(value), position, score, index) for index, (position, value, score)
        in enumerate(zip(positions, values, scores, strict=True))
        if (keep_first and index == 0) or (np.isfinite(value) and np.all(np.isfinite(score)))]
    rows.sort(key=lambda row: row[0], reverse=True)
    if not rows:
        return -1, 0, None
    return rows[0][3], len(rows), rows[0][:3]


@pytest.mark.parametrize("kind", ["ties", "mixed", "none", "position_nonfinite", "nan_incumbent", "negative_infinity_incumbent"])
def test_selection_keeps_original_eligibility_and_stable_ties(kind):
    positions = np.array([[.1, .2], [.3, .4], [.5, .6]])
    values, scores = np.array([2., 2., 1.]), np.ones([3, 2])
    keep_first = kind.endswith("incumbent")
    if kind == "mixed":
        values[0], scores[1, 0] = np.nan, np.inf
    elif kind == "none":
        values[:] = np.nan
    elif kind == "position_nonfinite":
        positions[0, 0] = np.nan
    elif kind == "nan_incumbent":
        values[0] = np.nan
    elif kind == "negative_infinity_incumbent":
        values[0] = -np.inf
    index, count, expected = _reference(positions, values, scores, keep_first=keep_first)

    @tf.function(input_signature=[tf.TensorSpec([3, 2], D), tf.TensorSpec([3], D),
        tf.TensorSpec([3, 2], D)], jit_compile=True, autograph=False)
    def compiled(positions, values, scores):
        return native.selection_numerics(positions, values, scores, keep_first=keep_first)

    result = compiled(positions, values, scores)
    assert int(result['index']) == index and int(result['finite_count']) == count
    if expected is not None:
        for field, value in zip(('value', 'position', 'score'), expected, strict=True):
            np.testing.assert_equal(result[field].numpy(), value)


@pytest.mark.parametrize("count", [0, 1, 3, 7])
def test_replay_evaluates_every_row_once_in_original_order(count):
    order = tf.Variable(0, dtype=tf.int64)
    positions = tf.reshape(tf.cast(tf.range(2 * count), D), [count, 2])

    def target(point):
        visited = order.assign(order * 17 + tf.cast(point[0], tf.int64) + 1)
        with tf.control_dependencies([visited]):
            value = tf.where(point[0] == 2., tf.constant(float('nan'), D), point[0])
            return value, tf.ones([2], D)

    result = candidate._replay_locator_candidates(target, positions)
    expected_order = 0
    for point in positions:
        expected_order = expected_order * 17 + int(point[0]) + 1
    assert int(order) == expected_order
    assert int(result['finite_count']) == count - int(count >= 2)
    assert int(result['index']) == count - 1


def _targets():
    precision = tf.constant([[2., .3], [.3, 3.]], D)

    def scalar(point):
        score = -tf.linalg.matvec(precision, point)
        return .5 * tf.reduce_sum(point * score), score

    def batch(points):
        scores = -points @ precision
        return .5 * tf.reduce_sum(points * scores, axis=1), scores

    return scalar, batch


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("orthogonal", [False, True])
def test_complete_search_keeps_cloud_order_values_scores_and_winner(previous, batched, orthogonal):
    scalar, batch = _targets()
    batch = batch if batched else None
    center, scale = tf.constant([.3, -.4], D), tf.constant([.5, 1.2], D)
    center_value, center_score = scalar(center)
    radius, seed = tf.constant(.35, D), tf.constant([2026, 734])
    count = 8
    cloud = (previous._orthogonal_antithetic_cloud if orthogonal else previous._antithetic_cloud)(
        count, 2, float(radius), tuple(seed.numpy().tolist()))
    positions = center[None, :] + cloud * scale[None, :]
    values, scores = previous._evaluate_cloud(scalar, positions, 2, batched_value_and_score_fn=batch)
    rows = np.concatenate([center[None, :], positions])
    index, finite_count, expected = _reference(rows, np.r_[center_value, values],
        np.concatenate([center_score[None, :], scores]), keep_first=True)
    arguments = (center, center_value, center_score, scale, radius, seed)
    result = candidate._search_exact_candidates(scalar, batch, *arguments,
        sample_count=count, orthogonal=orthogonal)
    assert int(result['index']) == index and int(result['finite_count']) == finite_count
    for name, value in zip(('value', 'position', 'score'), expected, strict=True):
        np.testing.assert_allclose(result[name], value, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(result['search_positions'], positions, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(result['search_values'], values, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(result['search_scores'], scores, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("kind", ["replay", "search"])
def test_compilation_graph_size_is_bounded(kind):
    scalar, batch = _targets()
    node_counts = []
    for count in (4, 8):
        if kind == 'replay':
            programs = [native.replay_program(scalar, count, 2, jit_compile=jit) for jit in (False, True)]
            arguments = (tf.reshape(tf.linspace(tf.constant(.1, D), tf.constant(1., D), 2 * count), [count, 2]),)
        else:
            programs = [native.search_program(scalar, batch, count, 2, False, jit_compile=jit) for jit in (False, True)]
            center = tf.constant([.4, .2], D)
            value, score = scalar(center)
            arguments = (center, value, score, tf.ones([2], D), tf.constant(.3, D), tf.constant([23, 45]))
        before, after = [program(*arguments) for program in programs]
        for name in before:
            np.testing.assert_allclose(after[name], before[name], atol=1e-10, rtol=1e-10)
        assert 'HloModule' in programs[1].experimental_get_compiler_ir(*arguments)(stage='hlo')
        counts = []
        for jit, program in zip((False, True), programs, strict=True):
            graph = program.get_concrete_function().graph.as_graph_def()
            nodes = list(graph.node) + [node for function in graph.library.function for node in function.node_def]
            assert not any(node.op in ('PyFunc', 'EagerPyFunc', 'PyFuncStateless') for node in nodes)
            if not jit:
                assert not any(function.attr['_XlaMustCompile'].b for function in graph.library.function
                    if '_XlaMustCompile' in function.attr)
            assert program.experimental_get_tracing_count() == 1
            counts.append(len(nodes))
        node_counts.append(counts)
    assert node_counts[0] == node_counts[1]


def test_selected_position_and_score_keep_the_gather_pullback():
    scalar, _ = _targets()
    replay = native.replay_program(scalar, 3, 2)

    @tf.function(input_signature=[tf.TensorSpec([3, 2], D)], jit_compile=True, autograph=False)
    def derivative(rows):
        with tf.GradientTape() as tape:
            tape.watch(rows)
            selected = replay(rows)
            objective = tf.reduce_sum(selected['position'] + selected['score'])
        return tape.gradient(objective, rows)

    rows = tf.constant([[.4, .5], [.1, .2], [.3, .4]], D)
    expected = np.zeros([3, 2])
    expected[1] = [1. - 2. - .3, 1. - 3. - .3]
    np.testing.assert_allclose(derivative(rows), expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("policy,budget", [('center_first', 4), ('center_first', 128), ('multistart', 1)])
def test_existing_sequential_consumer_preserves_complete_records(previous, policy, budget):
    scalar, batch = _targets()
    starts = [[.2, -.3]] if policy == 'center_first' else [[.2, -.3], [-.4, .1]]
    results = []
    for module in (previous, candidate):
        config = module.SequentialMapCovarianceConfig(locator_policy=policy, max_exact_evaluations=budget,
            locator_max_iterations=1, max_attempts=2, search_sample_count=8,
            regression_sample_count=16, terminal_sample_count=16, terminal_score_max_abs=1e-8,
            record_refinement_movement_diagnostics=True)
        results.append(module.estimate_sequential_map_covariance(scalar, starts,
            batched_value_and_score_fn=batch, config=config).payload())
    _compare(results[1], results[0])
