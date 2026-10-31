"""Complete original dense initializer records with identical frozen clouds."""

import ast
import contextlib
import dataclasses
import gc
import hashlib
import math
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import fixed_center_curvature as fixed
from bayesfilter.inference.batched_local_center import (
    BatchedLocalCenterConfig,
)
from bayesfilter.inference.dense_initializer_controller_tf import (
    DenseInitializerProgram,
)
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
from bayesfilter.inference.tensor_npz_archive import write_tensor_npz
from tests.test_filter_repair_batched_center_reuse import original_source
from tests.test_filter_repair_block_capture import stable_hlo
from tests.test_filter_repair_dense_initializer_cloud import SOURCE, SOURCE_SHA
from tests.test_filter_repair_dense_validated_fit import (
    _thresholds,
    normalized,
    original_fitter,
)
from tests.test_filter_repair_geometry_control import clean, save
from tests.test_filter_repair_quadratic_batches import _equal_records

D = tf.float64


def diagnostic_ready(value):
    if tf.is_tensor(value) or isinstance(value, np.ndarray):
        return diagnostic_ready(value.numpy().tolist() if tf.is_tensor(value) else value.tolist())
    if isinstance(value, dict):
        return {str(name): diagnostic_ready(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [diagnostic_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def original_initializer(configuration, thresholds, clouds=None):
    text = SOURCE.read_text()
    assert hashlib.sha256(text.encode()).hexdigest() == SOURCE_SHA
    tree = ast.parse(text)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    function.body = [node for node in function.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    loop = next(node for node in ast.walk(function) if isinstance(node, ast.For)
        and isinstance(node.target, ast.Tuple) and ast.unparse(node.target) == '(partition_index, rows)')
    position_index = next(index for index, node in enumerate(loop.body)
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'positions')
    if clouds is not None:
        loop.body = [ast.parse('offsets = clouds[attempt, partition_index, :rows]').body[0], *loop.body[position_index:]]
    config_call = next(node for node in ast.walk(function) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == 'BatchedLocalCenterConfig')
    jit = next(option for option in config_call.keywords if option.arg == 'jit_compile')
    assert ast.unparse(jit.value) == 'False'
    jit.value = ast.Constant(True)
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    excerpt = ast.unparse(module)
    locator_checkpoint, locator, compatibility = original_source('dense_controller_original_locator')
    fitter, hashes = original_fitter()
    namespace = {'Any': object, 'np': np, 'tf': tf, 'math': math, 'clouds': clouds,
        'BatchedLocalCenterConfig': locator.BatchedLocalCenterConfig,
        'locate_batched_local_center': locator.locate_batched_local_center,
        'fit_fixed_center_curvature': fitter.fit_fixed_center_curvature,
        'initializer_configurations': lambda _: (configuration, None, thresholds),
        'diagnostic_json_ready': diagnostic_ready, 'json_ready': diagnostic_ready,
        'file_hash': lambda path: hashlib.sha256(path.read_bytes()).hexdigest()}
    exec(compile(module, str(SOURCE) + ':frozen_cloud_controller', 'exec'), namespace)  # noqa: S102
    return namespace['initialize_dense_local'], {
        'source': str(SOURCE), 'source_sha256': SOURCE_SHA,
        'adapted_source': excerpt, 'adapted_source_sha256': hashlib.sha256(excerpt.encode()).hexdigest(),
        'locator_sources': locator_checkpoint.hashes(), 'fitter_sources': hashes,
        'GPU_locator_accounting_adaptation': compatibility,
        'jit_setting': 'Explicit policy migration: original False compared at True; no numerical claim about this flag.'}


def completed_record(raw, configuration, thresholds, rows, owner):
    return format_dense_initializer_result(raw, configuration, thresholds, rows,
        trace_count=owner.locator.compiled.experimental_get_tracing_count(), planned_rows=owner.planned_rows)


def normalize_result(result):
    result = diagnostic_ready(result)
    result.pop('artifacts', None)
    for record in result.get('attempts', []):
        if 'curvature' in record:
            record['curvature'] = normalized({'result': record['curvature']})['result']
    return result


@pytest.mark.parametrize('dimension', [1, 3])
@pytest.mark.parametrize('case', ['healthy', 'invalid_locator', 'invalid_cloud', 'score_veto', 'fit_rejected',
    'invalid_second_cloud', 'invalid_rank', 'overlap', 'moved_retry', 'exhausted'])
def test_complete_original_dense_attempt_controller(dimension, case, tmp_path, request, *, seeded=False):
    if dimension != 1 and case in ('moved_retry', 'exhausted'):
        pytest.skip('Recurrence branches use the explicit D1 polynomial target; D3 healthy remains separate.')
    rows = [3 * dimension] * 2 + [2 * dimension] * 2 + [2 * dimension + 1]
    locator = JointCenterLocatorConfig(max_iterations=4, max_objective_evaluations=40,
        gradient_tolerance=1e-8)
    configuration = PosteriorLocalInitializerConfig(locator_config=locator, max_curvature_attempts=2,
        training_rows_per_replicate=rows[0], selection_rows_per_replicate=rows[2],
        audit_rows=rows[-1], replicate_count=2, seed=(215, 91))
    thresholds = _thresholds(fixed, incomplete=case == 'fit_rejected')
    if case == 'invalid_rank':
        thresholds = dataclasses.replace(thresholds, principal_subspace_rank=dimension + 1)
    if case == 'exhausted':
        configuration = dataclasses.replace(configuration, max_curvature_attempts=1)
    clouds = tf.stack([tf.stack([tf.pad(.13 * tf.random.stateless_normal([n, dimension],
        [217 + attempt, index], dtype=D), [[0, max(rows) - n], [0, 0]])
        for index, n in enumerate(rows)]) for attempt in range(2)])
    if case in ('moved_retry', 'exhausted'):
        # f(x)=-10*x^2*(x-1)^2 + 3*x^2 - 2*x^3 has stationary local
        # maxima at zero and one, with f(1)>f(0). All scores are analytic.
        # The first exact cloud visits one; the second stays near that maximum.
        clouds = clouds * .01
        clouds = tf.tensor_scatter_nd_update(clouds, [[0, 0, 0, 0]], [tf.constant(1.25, D)])
    if case == 'invalid_second_cloud':
        clouds = tf.tensor_scatter_nd_update(clouds, [[0, 1, 0, 0]], [tf.constant(-1000., D)])
    if case == 'overlap':
        clouds = tf.tensor_scatter_nd_update(clouds, [[0, 4, 0]], clouds[0, 0, :1])
    clouds = clouds[:configuration.max_curvature_attempts]
    precision = tf.constant([[1.7]], D)
    if dimension == 3:
        # Exact one-factor covariance with loadings away from the constraint.
        loadings = tf.constant([.2, .35, -.15], D)
        marginal = tf.constant([.9, 1.2, 1.4], D)
        correlation = tf.linalg.diag(1. - loadings**2) + loadings[:, None] * loadings[None, :]
        precision = tf.linalg.inv(marginal[:, None] * correlation * marginal[None, :])
    calls = tf.Variable(0, dtype=tf.int64, trainable=False)
    count = tf.Variable(0, dtype=tf.int64, trainable=False)
    positions = tf.Variable(tf.zeros([1000, dimension], D), trainable=False)
    extents = tf.Variable(tf.zeros([1000], tf.int64), trainable=False)

    def callback(points):
        n = points.shape[0]
        index = count.assign_add(n) - n
        call = calls.assign_add(1) - 1
        positions.scatter_nd_update((index + tf.range(n, dtype=tf.int64))[:, None], points)
        extents.scatter_nd_update([[call]], [tf.cast(n, tf.int64)])
        score = -tf.linalg.matmul(points, precision, transpose_b=True)
        value = .5 * tf.reduce_sum(points * score, axis=1)
        if case in ('moved_retry', 'exhausted'):
            x = points[:, 0]
            value = -10. * x**2 * (x - 1.)**2 + 3. * x**2 - 2. * x**3
            score = (2. * x * (1. - x) * (10. * (2. * x - 1.) + 3.))[:, None]
        if case == 'score_veto':
            score, value = tf.ones_like(points), tf.reduce_sum(points, axis=1)
        valid = tf.ones([n], tf.bool)
        if case == 'invalid_locator' or (case == 'invalid_cloud' and n != 1):
            valid = tf.zeros_like(valid)
        if case == 'invalid_second_cloud':
            valid &= tf.reduce_all(points > -100., axis=1)
        return value, score, valid

    score_max = .1 if case == 'score_veto' else 1e-6
    locator_config = BatchedLocalCenterConfig(box_radius=configuration.locator_box_radius,
        trust_refinement_rounds=1, num_correction_pairs=locator.num_correction_pairs,
        max_iterations=locator.max_iterations, max_line_search_iterations=locator.max_line_search_iterations,
        gradient_tolerance=locator.gradient_tolerance,
        max_optimizer_callback_batches_per_round=locator.max_objective_evaluations)
    program_type = SeededDenseInitializerProgram if seeded else DenseInitializerProgram
    owner = program_type(callback, dimension, 2, rows[0], rows[2], rows[-1],
        locator_config=locator_config, thresholds=thresholds, max_attempts=configuration.max_curvature_attempts,
        max_exact_evaluations=configuration.max_exact_evaluations, center_score_max=score_max)
    controller = owner.controller if seeded else owner
    reports, hlos = [], []
    for iteration, shift in enumerate((0., .00001, 0.)):
        # Keep analytic stationary centers exact in recurrence probes; cloud
        # perturbations still verify changed operands and graph reuse.
        center_shift = 0. if case in ('moved_retry', 'exhausted') else shift
        center = tf.fill([dimension], tf.constant(center_shift, D))
        scale = .8 + tf.cast(tf.range(dimension), D) * .2 + shift
        offsets = clouds + shift
        if case in ('moved_retry', 'exhausted'):
            offsets = tf.tensor_scatter_nd_update(offsets, [[0, 0, 0, 0]], [1. / scale[0]])
        active_configuration = (dataclasses.replace(configuration,
            seed=(215 + iteration % 2, 91 - iteration % 2), curvature_radius=.13 + shift)
            if seeded else configuration)
        original, provenance = original_initializer(active_configuration, thresholds, None if seeded else offsets)
        radius = tf.constant(active_configuration.curvature_radius, D)
        operands = ((center, scale, tf.constant(active_configuration.seed, tf.int32), radius)
            if seeded else (center, scale, offsets))
        watched = (center, scale, radius) if seeded else (center, scale, offsets)
        root = tmp_path / f'original-{iteration}'
        root.mkdir()
        context = SimpleNamespace(root=root, boundary=lambda _: contextlib.nullcontext())
        model = SimpleNamespace(training_target=SimpleNamespace(parameter_dim=dimension,
            batch_value_score_and_validity=callback), coordinate_scale=scale, initial_position=center)
        recipe = SimpleNamespace(dense_center_score_max=score_max)
        calls.assign(0)
        count.assign(0)
        try:
            expected = original(model, recipe, context)
        except ValueError as error:
            if case not in ('invalid_rank', 'overlap'):
                raise
            expected = {'error': {'type': type(error).__name__, 'message': str(error)}}
        old_calls = {'calls': int(calls), 'rows': int(count), 'positions': clean(positions[:int(count)]),
            'extents': clean(extents[:int(calls)])}
        expected_archives = {}
        if (root / 'initializer_evaluations.npz').exists():
            with np.load(root / 'initializer_evaluations.npz') as handle:
                expected_archives = {name: handle[name].copy() for name in handle.files}
        calls.assign(0)
        count.assign(0)
        with tf.GradientTape() as tape:
            tape.watch(watched)
            raw = owner(*operands)
            value = tf.reduce_sum(raw['initial_output_scale_log'])
        derivatives = tape.gradient(value, watched)
        try:
            actual, archives = completed_record(raw, active_configuration, thresholds, rows, controller)
        except ValueError as error:
            if case not in ('invalid_rank', 'overlap'):
                raise
            actual, archives = {'error': {'type': type(error).__name__, 'message': str(error)}}, {}
        if archives:
            candidate_path = tmp_path / f'candidate-{iteration}.npz'
            with candidate_path.open('xb') as stream:
                write_tensor_npz(stream, archives)
            with np.load(candidate_path) as archive:
                assert set(archive.files) == set(expected_archives)
                for name in archive.files:
                    np.testing.assert_allclose(archive[name], expected_archives[name], atol=1e-10, rtol=1e-10)
                    np.testing.assert_array_equal(archive[name], archives[name])
        actual_calls = {'calls': int(calls), 'rows': int(count), 'positions': clean(positions[:int(count)]),
            'extents': clean(extents[:int(calls)])}
        reports.append({'actual': normalize_result(actual), 'original': normalize_result(expected),
            'actual_archives': clean(archives), 'original_archives': clean(expected_archives),
            'actual_calls': actual_calls, 'original_calls': old_calls,
            'frozen_derivatives': [value is None for value in derivatives],
            'raw_status': int(raw['status_code']), 'raw_attempt_count': int(raw['attempt_count'])})
        hlos.append(stable_hlo(owner.compiled.experimental_get_compiler_ir(*operands)(stage='hlo')))
    traces = owner.compiled.experimental_get_tracing_count()
    refs = {'owner': weakref.ref(owner), 'locator': weakref.ref(controller.locator),
        'callback': weakref.ref(callback), 'attempt': weakref.ref(controller.attempt),
        'locator_graph': weakref.ref(controller.locator.compiled.get_concrete_function().graph),
        'attempt_graph': weakref.ref(controller.attempt.get_concrete_function().graph),
        'graph': weakref.ref(owner.compiled.get_concrete_function().graph)}
    if seeded:
        refs.update(controller=weakref.ref(controller), cloud_design=weakref.ref(owner.cloud_design),
            controller_graph=weakref.ref(controller.compiled.get_concrete_function().graph),
            rng_graph=weakref.ref(owner.cloud_design.get_concrete_function().graph))
    del owner, controller, callback, original, model
    gc.collect()
    released = {name: value() is None for name, value in refs.items()}
    report = {'dimension': dimension, 'case': case, 'records': reports,
        'provenance': provenance, 'trace_count': traces, 'hlo_unchanged': len(set(hlos)) == 1,
        'python_released': released, 'seeded': seeded, 'nonclaims': ['No actual-DZ5 qualification.'] if seeded else
            ['Frozen clouds; no seeded or actual-DZ5 qualification.']}
    save(request, f'dense-controller-{dimension}-{case}{"-seeded" if seeded else ""}.json', report)
    for row in reports:
        _equal_records(row['actual'], row['original'])
        _equal_records(row['actual_archives'], row['original_archives'])
        _equal_records(row['actual_calls'], row['original_calls'])
        assert all(row['frozen_derivatives'])
        if case == 'moved_retry':
            assert row['raw_attempt_count'] == 2 and row['actual']['passed']
            assert row['actual']['attempts'][0]['curvature_center_moved']
        if case == 'exhausted':
            assert row['raw_attempt_count'] == 1 and row['raw_status'] == 4
        if case == 'invalid_second_cloud':
            assert len(row['actual_archives']) == 8 and row['raw_status'] == 3
        if case in ('invalid_rank', 'overlap'):
            assert 'error' in row['actual'] and not row['actual_archives']
    if seeded:
        assert 'tf.random.stateless_normal' in provenance['adapted_source']
        assert 'clouds[attempt' not in provenance['adapted_source']
        _equal_records(reports[0], reports[2])
    assert traces == 1 and report['hlo_unchanged'] and all(released.values())


@pytest.mark.parametrize('dimension', [1, 3])
@pytest.mark.parametrize('case', ['healthy', 'invalid_locator', 'invalid_cloud', 'score_veto', 'fit_rejected'])
def test_complete_original_seeded_controller(dimension, case, tmp_path, request):
    test_complete_original_dense_attempt_controller(dimension, case, tmp_path, request, seeded=True)
