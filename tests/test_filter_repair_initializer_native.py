"""Independent original initializer payloads, exact calls and enclosing XLA."""

import dataclasses
import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.quadratic_geometry import LowRankSPDQuadraticGeometryConfig
from bayesfilter.inference.quadratic_geometry_full_tf import prepare_geometry_inputs
from bayesfilter.inference.quadratic_initializer_report import initializer_result
from bayesfilter.inference.quadratic_initializer_tf import (
    make_quadratic_initializer_program,
)
from bayesfilter.inference.quadratic_map_covariance import (
    IterativeQuadraticMapCovarianceConfig,
    QuadraticMapCovarianceLocatorConfig,
    QuadraticMapCovarianceMassConfig,
)
from bayesfilter.ops.geometry_random_tf import _draw_kernel
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_geometry_control import clean, save
from tests.test_filter_repair_joint_center_native import original_source
from tests.test_filter_repair_quadratic_batches import _equal_records

D = tf.float64


def fixture(dimension, case, batched):
    calls, rows = tf.Variable(0, dtype=tf.int64), tf.Variable(0, dtype=tf.int64)
    positions = tf.Variable(tf.zeros([2048, dimension], D))
    extents = tf.Variable(tf.zeros([2048], tf.int64))

    def compute(cloud):
        extent = cloud.shape[0]
        index, start = calls.assign_add(1) - 1, rows.assign_add(extent) - extent
        logs = (positions.scatter_nd_update((start + tf.range(extent, dtype=tf.int64))[:, None], cloud),
                extents.scatter_nd_update(index[None, None], tf.constant([extent], tf.int64)))
        with tf.control_dependencies(logs):
            delta = cloud - .34
            scores = -1.7 * delta
            values = .5 * tf.reduce_sum(delta * scores, 1)
        if case == 'nonfinite':
            values = tf.fill([extent], tf.constant(float('nan'), D))
        if case in ('bad_center', 'bad_geometry'):
            values = tf.where(start == (1 if case == 'bad_center' else 2),
                              tf.constant(float('nan'), D), values)
        return values, scores

    def scalar(point):
        values, scores = compute(point[None])
        return values[0], scores[0]

    def reset():
        calls.assign(0)
        rows.assign(0)
        positions.assign(tf.zeros_like(positions))
        extents.assign(tf.zeros_like(extents))

    def log():
        return clean({'calls': calls.read_value(), 'rows': rows.read_value(),
                      'positions': positions[:int(rows)], 'extents': extents[:int(calls)]})

    geometry = LowRankSPDQuadraticGeometryConfig(rank=1, min_samples_per_parameter=1,
        sample_count=2 if case == 'insufficient' else 16, pilot_direction_count=6,
        trust_radius=.15, pilot_radius=.05, holdout_fraction=.25,
        eigenvalue_floor=.1, max_condition_number=100., holdout_rmse_abs_tolerance=1.,
        constrain_center_refinement_to_trust_region=True, seed=(37, 41))
    locator = QuadraticMapCovarianceLocatorConfig(enabled=case == 'enabled', max_iterations=6)
    mass = QuadraticMapCovarianceMassConfig(jitter=0., eigenvalue_floor=.1,
        max_condition_number=100., dense=case != 'diagonal')
    iterative = (None if case == 'single' else IterativeQuadraticMapCovarianceConfig(
        max_refinement_steps=1 if case == 'budget' else 6, terminal_score_max_abs=1e-7))
    initial = tf.fill([dimension], tf.constant(.34 if case in ('single', 'diagonal') else .1, D))
    scale = tf.fill([dimension], tf.constant(.9, D))
    return scalar, compute if batched else None, locator, geometry, mass, iterative, initial, scale, reset, log


def original(callback, batch, configs, inputs, monkeypatch):
    checkpoint = FrozenCheckpoint('3582b4ac', 'initializer_native_original')
    wrapper = checkpoint.load('bayesfilter.inference.quadratic_map_covariance')
    geometry = checkpoint.load('bayesfilter.inference.quadratic_geometry')
    _, joint, compatibility = original_source('initializer_joint_original')
    wrapper.locate_joint_center = joint.locate_joint_center
    locator_config, geometry_config, mass_config, iterative_config = configs
    initial, scale, directions, offsets, seed = inputs

    class FrozenNumpy:
        random = SimpleNamespace(default_rng=lambda ignored: SimpleNamespace(
            normal=lambda **kw: directions.numpy().copy(),
            permutation=lambda count: _draw_kernel('permutation', (count,))(seed).numpy()))

        def __getattr__(self, name):
            return getattr(np, name)

    options = {'value_and_score_fn': callback, 'batched_value_and_score_fn': batch,
        'initial_position': initial.numpy(), 'scale': scale.numpy(),
        'locator_config': wrapper.QuadraticMapCovarianceLocatorConfig(**dataclasses.asdict(locator_config)),
        'quadratic_config': geometry.LowRankSPDQuadraticGeometryConfig(**dataclasses.asdict(geometry_config)),
        'mass_config': wrapper.QuadraticMapCovarianceMassConfig(**dataclasses.asdict(mass_config))}
    events = []
    if iterative_config is not None:
        options.update(iterative_config=wrapper.IterativeQuadraticMapCovarianceConfig(**dataclasses.asdict(iterative_config)),
            fit_start_callback=lambda i, center: events.append(('start', i, clean(center))),
            iteration_callback=lambda row: events.append(('iteration', clean(row))))
    name = 'estimate_iterative_quadratic_map_covariance' if iterative_config is not None else 'estimate_quadratic_map_covariance'
    original_json = wrapper._json_ready

    def tensor_record(value):
        # The old wrapper stringifies its own TF mass arrays, losing digits.
        # Materialize those existing results only; no numerical operation changes.
        return original_json(value.numpy() if tf.is_tensor(value) else value)

    with monkeypatch.context() as patch:
        patch.setattr(geometry, 'np', FrozenNumpy())
        patch.setattr(geometry, '_sample_trust_ball', lambda *args, **kw: offsets.numpy().copy())
        patch.setattr(wrapper, '_json_ready', tensor_record)
        result = getattr(wrapper, name)(**options)
        payload = clean(result.payload(include_arrays=True))
    return payload, clean(events), checkpoint.hashes(), compatibility


def comparison(record):
    """Normalize only approved RNG metadata, byte hashes and historical labels."""
    if isinstance(record, dict):
        omitted = ('random_stream', 'artifact_hash')
        # XLA was explicitly promoted during the repair. The old locator
        # report had no JIT field; assert the new True value separately.
        if record.get('schema') == 'bayesfilter.quadratic_map_covariance.locator.v1':
            omitted += ('jit_compile',)
        return {key: comparison(value) for key, value in record.items() if key not in omitted}
    if isinstance(record, (list, tuple)):
        return [comparison(value) for value in record]
    if isinstance(record, str) and record == 'scalar_value_and_score_loop':
        return 'tensorflow_scalar_row_loop'
    return record


@pytest.mark.parametrize('dimension', [1, 3])
@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('case', ['iterative', 'single', 'enabled', 'budget', 'diagonal',
                                 'nonfinite', 'bad_center', 'bad_geometry', 'insufficient'])
def test_initializer_native_original_records(dimension, batched, case, monkeypatch, request):
    callback, batch, locator, geometry, mass, iterative, initial, scale, reset, log = fixture(dimension, case, batched)
    inputs = (initial, scale, *prepare_geometry_inputs(dimension, geometry))
    configs = locator, geometry, mass, iterative
    expected, expected_events, hashes, compatibility = original(callback, batch, configs, inputs, monkeypatch)
    expected_calls = log()
    reset()
    program = make_quadratic_initializer_program(callback, dimension, locator, geometry, mass,
        iterative_config=iterative, batched_callback=batch)
    events = []
    with program.invocation_lock:
        raw = program(*inputs)
        result = initializer_result(raw, initial, scale, locator, geometry, mass,
            iterative_config=iterative, batched=batched,
            fit_start_callback=lambda i, center: events.append(('start', i, clean(center))),
            iteration_callback=lambda row: events.append(('iteration', clean(row))))
    actual = clean(result.payload(include_arrays=True))
    actual_calls, actual_events = log(), clean(events)
    save(request, f'initializer-native-{dimension}-{batched}-{case}.json', {
        'actual': actual, 'expected': expected, 'actual_events': actual_events, 'expected_events': expected_events,
        'actual_calls': actual_calls, 'expected_calls': expected_calls,
        'original_sources': hashes, 'joint_reference_gpu_compatibility': compatibility})
    _equal_records(comparison(actual), comparison(expected))
    _equal_records(comparison(actual_events), comparison(expected_events))
    _equal_records(actual_calls, expected_calls)
    assert program.experimental_get_tracing_count() == 1
    assert program.get_concrete_function().function_def.attr['_XlaMustCompile'].b
    if case != 'nonfinite':
        assert actual['locator_diagnostics']['jit_compile'] is True


def test_initializer_roundoff_attribution(monkeypatch, request):
    """Compare original and native geometry on the same observed centers."""
    from bayesfilter.inference.quadratic_geometry_full_report import geometry_result
    from bayesfilter.inference.quadratic_geometry_full_tf import make_geometry_program
    from tests.test_filter_repair_geometry_full import original as original_geometry

    callback, _batch, _locator, cfg, _mass, _iterative, _initial, scale, reset, log = fixture(1, 'iterative', False)
    prepared = prepare_geometry_inputs(1, cfg)
    native = make_geometry_program(callback, 1, cfg)
    rows = []
    for center in (.1, .23499999999995466, .2349999999999547, .34, .3400000000000001):
        point = tf.constant([center], D)
        args = (point, scale, *prepared)
        reset()
        expected, hashes = original_geometry(callback, None, cfg, args, monkeypatch)
        original_calls = log()
        reset()
        raw = native(*args)
        actual = geometry_result(raw, point, scale, cfg)
        rows.append({'center_hex': float(center).hex(), 'expected': expected,
            'actual': clean(actual.payload(include_arrays=True)), 'original_calls': original_calls,
            'actual_calls': log(), 'raw': clean(raw), 'original_sources': hashes})
    save(request, 'initializer-roundoff-attribution.json', {'comparisons': rows,
         'role': 'explanatory attribution; no tolerance or status waiver'})


def test_initializer_affine_boundary_attribution(monkeypatch, request):
    """Isolate former NumPy multiply/add boundaries without changing runtime."""
    from bayesfilter.inference import (
        quadratic_geometry_control_tf as control,
    )
    from bayesfilter.inference import (
        quadratic_geometry_fit_tf as fitting,
    )
    from bayesfilter.inference import (
        quadratic_geometry_full_tf as complete,
    )
    from bayesfilter.inference import (
        quadratic_geometry_prepare_tf as preparation,
    )
    from bayesfilter.inference.joint_center_tf import rounded_affine_position
    from bayesfilter.inference.quadratic_geometry_full_report import geometry_result
    from tests.test_filter_repair_geometry_full import original as original_geometry

    def adapted(function, before, after):
        source = inspect.getsource(function)
        assert source.count(before) == 1
        scope = {**function.__globals__, 'rounded_affine_position': rounded_affine_position}
        exec(compile(source.replace(before, after), 'diagnostic_rounded_boundary', 'exec'), scope)  # noqa: S102
        return scope[function.__name__]

    design = adapted(preparation.make_geometry_design_program,
        'positions = center[None] + offsets * scale[None]',
        'positions = rounded_affine_position(center[None], scale[None], offsets)')
    refinement = adapted(control.make_center_refinement_program,
        'refined = center + step["step"] * scale',
        'refined = rounded_affine_position(center, scale, step["step"])')
    callback, _batch, locator, cfg, mass, iterative, initial, scale, reset, _log = fixture(1, 'iterative', False)
    prepared = prepare_geometry_inputs(1, cfg)
    records = {}
    for label in ('design', 'refinement', 'both'):
        with monkeypatch.context() as patch:
            if label in ('design', 'both'):
                patch.setattr(complete, 'make_geometry_design_program', design)
            if label in ('refinement', 'both'):
                patch.setattr(fitting, 'make_center_refinement_program', refinement)
            program = complete.make_geometry_program(callback, 1, cfg)
            rows = []
            for center in (.1, .23499999999995466, .2349999999999547, .34, .3400000000000001):
                point = tf.constant([center], D)
                args = (point, scale, *prepared)
                reset()
                expected, hashes = original_geometry(callback, None, cfg, args, monkeypatch)
                reset()
                raw = program(*args)
                actual = geometry_result(raw, point, scale, cfg)
                rows.append({'center': center, 'expected': expected, 'actual': clean(actual.payload(include_arrays=True)),
                             'raw': clean(raw), 'original_sources': hashes})
            reset()
            enclosing = make_quadratic_initializer_program(callback, 1, locator, cfg, mass, iterative_config=iterative)
            raw = enclosing(initial, scale, *prepared)
            actual = initializer_result(raw, initial, scale, locator, cfg, mass, iterative_config=iterative)
            records[label] = {'same_center': rows, 'complete': clean(actual.payload(include_arrays=True))}
    save(request, 'initializer-affine-boundary-attribution.json', records)


def prior_wrapper(callback, batch, configs, inputs, monkeypatch):
    """Pinned pre-enclosure controller with its same qualified geometry code."""
    checkpoint = FrozenCheckpoint('dba39e048', 'initializer_controller_mechanism')
    wrapper = checkpoint.load('bayesfilter.inference.quadratic_map_covariance')
    geometry = checkpoint.load('bayesfilter.inference.quadratic_geometry_full_tf')
    locator_config, geometry_config, mass_config, iterative_config = configs
    events = []
    options = {'value_and_score_fn': callback, 'initial_position': inputs[0],
        'scale': inputs[1], 'batched_value_and_score_fn': batch,
        'locator_config': locator_config, 'quadratic_config': geometry_config, 'mass_config': mass_config}
    if iterative_config is not None:
        options.update(iterative_config=iterative_config,
            fit_start_callback=lambda i, center: events.append(('start', i, clean(center))),
            iteration_callback=lambda row: events.append(('iteration', clean(row))))
    name = 'estimate_iterative_quadratic_map_covariance' if iterative_config is not None else 'estimate_quadratic_map_covariance'
    with monkeypatch.context() as patch:
        patch.setattr(geometry, 'prepare_geometry_inputs', lambda *args: inputs[2:])
        result = getattr(wrapper, name)(**options)
    return clean(result.payload(include_arrays=True)), clean(events), checkpoint.hashes()


@pytest.mark.parametrize('dimension', [1, 3])
@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('case', ['iterative', 'single', 'enabled', 'budget', 'diagonal',
                                 'nonfinite', 'bad_center', 'bad_geometry', 'insufficient'])
def test_initializer_native_controller_records(dimension, batched, case, monkeypatch, request):
    callback, batch, locator, geometry, mass, iterative, initial, scale, reset, log = fixture(dimension, case, batched)
    inputs = (initial, scale, *prepare_geometry_inputs(dimension, geometry))
    expected, expected_events, hashes = prior_wrapper(callback, batch, (locator, geometry, mass, iterative), inputs, monkeypatch)
    expected_calls = log()
    reset()
    program = make_quadratic_initializer_program(callback, dimension, locator, geometry, mass,
        iterative_config=iterative, batched_callback=batch)
    raw = program(*inputs)
    events = []
    actual = initializer_result(raw, initial, scale, locator, geometry, mass, iterative_config=iterative,
        batched=batched, fit_start_callback=lambda i, center: events.append(('start', i, clean(center))),
        iteration_callback=lambda row: events.append(('iteration', clean(row))))
    actual_payload, actual_events, actual_calls = clean(actual.payload(include_arrays=True)), clean(events), log()
    save(request, f'initializer-controller-{dimension}-{batched}-{case}.json', {
        'actual': actual_payload, 'expected': expected, 'actual_events': actual_events, 'expected_events': expected_events,
        'actual_calls': actual_calls, 'expected_calls': expected_calls, 'prior_sources': hashes,
        'classification': 'pinned dba39e048 controller comparison; original3582b4ac equivalence is a separate gate'})
    _equal_records(comparison(actual_payload), comparison(expected))
    _equal_records(comparison(actual_events), comparison(expected_events))
    _equal_records(actual_calls, expected_calls)


@pytest.mark.parametrize('case', ['shape', 'exception', 'nonfinite'])
def test_initializer_native_typed_target_failures(case, monkeypatch, request):
    dimension = 3
    _callback, _batch, locator, geometry, mass, iterative, initial, scale, _reset, _log = fixture(dimension, 'iterative', False)

    def callback(point):
        if case == 'exception':
            raise ValueError('synthetic construction failure')
        return tf.constant(float('nan') if case == 'nonfinite' else 1., D), point[:1] if case == 'shape' else point

    inputs = (initial, scale, *prepare_geometry_inputs(dimension, geometry))
    expected, _, hashes, _ = original(callback, None, (locator, geometry, mass, iterative), inputs, monkeypatch)
    program = make_quadratic_initializer_program(callback, dimension, locator, geometry, mass, iterative_config=iterative)
    raw = program(*inputs)
    actual = initializer_result(raw, initial, scale, locator, geometry, mass, iterative_config=iterative)
    actual_payload = clean(actual.payload(include_arrays=True))
    save(request, f'initializer-typed-{case}.json', {'actual': actual_payload, 'expected': expected,
        'original_sources': hashes, 'fit_count': int(raw['fit_count'])})
    _equal_records(comparison(actual_payload), comparison(expected))
    assert int(raw['fit_count']) == 0


def test_initializer_native_reuses_changed_operands(monkeypatch, request):
    callback, batch, locator, geometry, mass, iterative, initial, scale, reset, log = fixture(1, 'budget', True)
    inputs = (initial, scale, *prepare_geometry_inputs(1, geometry))
    program = make_quadratic_initializer_program(callback, 1, locator, geometry, mass,
        iterative_config=iterative, batched_callback=batch)
    records, hlo = [], []
    for shift in (0., .03):
        args = (initial + shift, scale + shift, *inputs[2:])
        reset()
        expected, _, hashes = prior_wrapper(callback, batch, (locator, geometry, mass, iterative), args, monkeypatch)
        expected_calls = log()
        reset()
        raw = program(*args)
        actual = initializer_result(raw, args[0], args[1], locator, geometry, mass,
                                    iterative_config=iterative, batched=True)
        actual_payload = clean(actual.payload(include_arrays=True))
        _equal_records(comparison(actual_payload), comparison(expected))
        _equal_records(log(), expected_calls)
        records.append({'actual': actual_payload, 'expected': expected, 'prior_sources': hashes})
        hlo.append(program.experimental_get_compiler_ir(*args)(stage='hlo'))
    graph = program.get_concrete_function().graph.as_graph_def()
    operations = {node.op for nodes in (graph.node, *(fn.node_def for fn in graph.library.function)) for node in nodes}
    assert not operations & {'PyFunc', 'EagerPyFunc', 'PyFuncStateless'}
    assert 'While' in operations or 'StatelessWhile' in operations
    assert hlo[0] == hlo[1]
    assert program.experimental_get_tracing_count() == 1
    assert records[0]['actual'] != records[1]['actual']
    save(request, 'initializer-controller-reuse.json', {'records': records, 'hlo': hlo[0]})
