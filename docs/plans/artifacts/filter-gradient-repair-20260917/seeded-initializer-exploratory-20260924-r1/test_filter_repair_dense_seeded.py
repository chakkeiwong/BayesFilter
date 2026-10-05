"""Full original seeded dense initializer versus one enclosing XLA program."""

import ast
import contextlib
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import fixed_center_curvature as fixed
from bayesfilter.inference.batched_local_center import BatchedLocalCenterConfig
from bayesfilter.inference.dense_initializer_reporting import (
    format_dense_initializer_result,
)
from bayesfilter.inference.dense_initializer_seeded_tf import (
    SeededDenseInitializerProgram,
)
from bayesfilter.inference.joint_center import JointCenterLocatorConfig
from bayesfilter.inference.posterior_local_initializer import (
    PosteriorLocalInitializerConfig,
)
from tests.test_filter_repair_block_capture import stable_hlo
from tests.test_filter_repair_dense_controller import diagnostic_ready
from tests.test_filter_repair_dense_initializer_cloud import SOURCE, SOURCE_SHA
from tests.test_filter_repair_dense_validated_fit import (
    _thresholds,
    original_fitter,
)
from tests.test_filter_repair_geometry_control import clean, save
from tests.test_filter_repair_quadratic_batches import _equal_records

D = tf.float64


def _original_seeded_initializer(configuration, thresholds):
    source = SOURCE.read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == SOURCE_SHA
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    function.body = [node for node in function.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    config_call = next(node for node in ast.walk(function) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == 'BatchedLocalCenterConfig')
    jit = next(option for option in config_call.keywords if option.arg == 'jit_compile')
    assert ast.unparse(jit.value) == 'False'
    jit.value = ast.Constant(True)
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    locator_checkpoint, locator, _ = __import__(
        'tests.test_filter_repair_batched_center_reuse', fromlist=['original_source']).original_source(
            'dense_seeded_original_locator')
    fitter, hashes = original_fitter()
    namespace = {
        'Any': object, 'np': np, 'tf': tf, 'math': __import__('math'),
        'BatchedLocalCenterConfig': locator.BatchedLocalCenterConfig,
        'locate_batched_local_center': locator.locate_batched_local_center,
        'fit_fixed_center_curvature': fitter.fit_fixed_center_curvature,
        'initializer_configurations': lambda _: (configuration, None, thresholds),
        'diagnostic_json_ready': diagnostic_ready, 'json_ready': diagnostic_ready,
        'file_hash': lambda path: hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    exec(compile(module, str(SOURCE) + ':seeded_original', 'exec'), namespace)  # noqa: S102
    return namespace['initialize_dense_local'], {
        'source': str(SOURCE), 'source_sha256': SOURCE_SHA,
        'adapted_source': ast.unparse(module),
        'locator_sources': locator_checkpoint.hashes(), 'fitter_sources': hashes,
        'jit_setting': 'Explicit policy migration: original False compared at True.',
    }


@pytest.mark.parametrize('dimension,rows', [
    (1, [3, 3, 2, 2, 3]), (3, [9, 9, 6, 6, 7]),
    (23, [68, 68, 46, 46, 46]),
])
def test_seeded_enclosing_program_preserves_full_records(dimension, rows, request, tmp_path):
    locator = JointCenterLocatorConfig(max_iterations=4, max_objective_evaluations=40,
        gradient_tolerance=1e-8)
    configuration = PosteriorLocalInitializerConfig(locator_config=locator, max_curvature_attempts=1,
        training_rows_per_replicate=rows[0], selection_rows_per_replicate=rows[2],
        audit_rows=rows[-1], replicate_count=2, seed=(215, 91), curvature_radius=.13)
    thresholds = _thresholds(fixed)
    scale = tf.ones([dimension], D)
    center = tf.zeros([dimension], D)
    precision = tf.eye(dimension, dtype=D)
    calls = tf.Variable(0, dtype=tf.int64, trainable=False)
    count = tf.Variable(0, dtype=tf.int64, trainable=False)
    positions = tf.Variable(tf.zeros([max(rows) * 20, dimension], D), trainable=False)
    extents = tf.Variable(tf.zeros([20], tf.int64), trainable=False)

    def callback(points):
        n = points.shape[0]
        index = count.assign_add(n) - n
        call = calls.assign_add(1) - 1
        positions.scatter_nd_update((index + tf.range(n, dtype=tf.int64))[:, None], points)
        extents.scatter_nd_update([[call]], [tf.cast(n, tf.int64)])
        score = -tf.linalg.matmul(points, precision, transpose_b=True)
        value = .5 * tf.reduce_sum(points * score, axis=1)
        return value, score, tf.ones([n], tf.bool)

    locator_config = BatchedLocalCenterConfig(box_radius=configuration.locator_box_radius,
        trust_refinement_rounds=1, num_correction_pairs=locator.num_correction_pairs,
        max_iterations=locator.max_iterations, max_line_search_iterations=locator.max_line_search_iterations,
        gradient_tolerance=locator.gradient_tolerance,
        max_optimizer_callback_batches_per_round=locator.max_objective_evaluations)
    owner = SeededDenseInitializerProgram(callback, dimension, 2, rows[0], rows[2], rows[-1],
        locator_config=locator_config, thresholds=thresholds, max_attempts=1,
        max_exact_evaluations=configuration.max_exact_evaluations,
        center_score_max=.02, factor_max=configuration.factor_max,
        dense_eigenvalue_floor=configuration.dense_eigenvalue_floor,
        max_condition_number=configuration.max_condition_number,
        shrinkage_weights=configuration.shrinkage_weights,
        structured_target_family=configuration.structured_target_family)
    original, provenance = _original_seeded_initializer(configuration, thresholds)
    root = tmp_path / 'original'
    root.mkdir()
    context = SimpleNamespace(root=root, boundary=lambda _: contextlib.nullcontext())
    model = SimpleNamespace(training_target=SimpleNamespace(parameter_dim=dimension,
        batch_value_score_and_validity=callback), coordinate_scale=scale,
        initial_position=center)
    recipe = SimpleNamespace(dense_center_score_max=.02)
    expected = original(model, recipe, context)
    expected_calls = {'calls': int(calls), 'rows': int(count),
        'positions': clean(positions[:int(count)]), 'extents': clean(extents[:int(calls)])}
    calls.assign(0)
    count.assign(0)
    raw = owner(center, scale, tf.constant(configuration.seed, tf.int32),
        tf.constant(configuration.curvature_radius, D))
    actual, archives = format_dense_initializer_result(raw, configuration, thresholds, rows,
        trace_count=owner.controller.locator.compiled.experimental_get_tracing_count(),
        planned_rows=owner.planned_rows)
    actual_calls = {'calls': int(calls), 'rows': int(count),
        'positions': clean(positions[:int(count)]), 'extents': clean(extents[:int(calls)])}
    normalized_expected = diagnostic_ready(expected)
    normalized_expected.pop('artifacts', None)
    normalized_actual = diagnostic_ready(actual)
    normalized_actual.pop('artifacts', None)
    _equal_records(normalized_actual, normalized_expected)
    _equal_records(actual_calls, expected_calls)
    assert archives and normalized_actual['passed']
    assert owner.compiled.experimental_get_tracing_count() == 1
    hlo1 = stable_hlo(owner.compiled.experimental_get_compiler_ir(
        center, scale, tf.constant(configuration.seed, tf.int32),
        tf.constant(configuration.curvature_radius, D))(stage='hlo'))
    changed = owner(center + .001, scale, tf.constant([317, -59], tf.int32), tf.constant(.21, D))
    hlo2 = stable_hlo(owner.compiled.experimental_get_compiler_ir(
        center + .001, scale, tf.constant([317, -59], tf.int32), tf.constant(.21, D))(stage='hlo'))
    assert hlo1 == hlo2 and int(changed['attempt_count']) == 1
    save(request, f'dense-seeded-{dimension}.json', {
        'dimension': dimension, 'rows': rows, 'stream': owner.stream_id,
        'provenance': provenance, 'actual': normalized_actual,
        'expected': normalized_expected, 'callback': actual_calls,
        'trace_count': owner.compiled.experimental_get_tracing_count(),
        'hlo_unchanged': hlo1 == hlo2,
        'nonclaims': ['Seeded full-controller parity only; actual DZ5 target remains open.'],
    })
