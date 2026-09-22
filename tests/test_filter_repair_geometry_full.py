"""Complete original geometry records, exact calls and enclosing XLA checks."""

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.quadratic_geometry import (
    LowRankSPDQuadraticGeometryConfig,
    fit_low_rank_spd_quadratic_geometry,
)
from bayesfilter.inference.quadratic_geometry_fit_report import geometry_fit_payload
from bayesfilter.inference.quadratic_geometry_full_report import geometry_result
from bayesfilter.inference.quadratic_geometry_full_tf import (
    geometry_extents,
    make_geometry_program,
    prepare_geometry_inputs,
)
from bayesfilter.ops.geometry_random_tf import GeometryTensorStream, _draw_kernel
from tests.test_filter_repair_geometry_active_pilot import stable_pilot_hlo
from tests.test_filter_repair_geometry_control import clean, save, source
from tests.test_filter_repair_quadratic_batches import _equal_records

D = tf.float64


def fixture(dimension, batched, case='gaussian', samples=16):
    cfg = LowRankSPDQuadraticGeometryConfig(rank=max(1, dimension - 1), sample_count=samples,
        min_samples_per_parameter=1, pilot_direction_count=6,
        holdout_fraction=0. if case == 'no_holdout' else .25,
        holdout_rmse_abs_tolerance=1e-14 if case == 'holdout_reject' else .05,
        holdout_rmse_rel_tolerance=1e-14 if case == 'holdout_reject' else .1,
        constrain_center_refinement_to_trust_region=case != 'unconstrained')
    rank, _, _, directions = geometry_extents(dimension, cfg)
    raw = np.random.default_rng(507 + dimension).normal(size=(directions, dimension))
    if case == 'partial_directions':
        raw[::3] = 0.
    if case == 'no_directions':
        raw[:] = 0.
    if case == 'unresolved_pilot':
        raw[1:] = 0.
    offsets = np.random.default_rng(641 + dimension).normal(size=(samples, dimension)) * .2
    if case == 'constant':
        offsets[:] = 0.
    center = tf.cast(tf.range(dimension), D) * .01
    scale = tf.cast(tf.range(dimension), D) * .1 + .8
    maximum = 1 + 2 * directions + samples + 2
    with tf.device(center.device):
        rows = tf.Variable(0, dtype=tf.int64)
        calls = tf.Variable(0, dtype=tf.int64)
        positions = tf.Variable(tf.zeros([maximum, dimension], D))
        extents = tf.Variable(tf.zeros([maximum], tf.int64))

    def compute(cloud, batch):
        extent = cloud.shape[0]
        call_index = calls.assign_add(1) - 1
        start = rows.assign_add(extent) - extent
        extents.scatter_nd_update(call_index[None, None], tf.constant([extent], tf.int64))
        positions.scatter_nd_update((start + tf.range(extent, dtype=tf.int64))[:, None], cloud)
        delta = cloud - .13
        score = -delta * (tf.cast(tf.range(dimension), D) + 2.)
        value = .5 * tf.reduce_sum(delta * score, axis=1)
        if case in ('nonquadratic', 'holdout_reject'):
            value -= .05 * tf.reduce_sum(delta ** 4, axis=1)
            score -= .2 * delta ** 3
        if case == 'coupled' and batch:
            average = tf.reduce_sum(cloud, axis=0) / tf.cast(max(1, extent), D)
            score -= .03 * average[None] ** 3
            value -= .03 * tf.reduce_sum(cloud * average[None] ** 3, axis=1)
        if case == 'constant':
            value, score = tf.zeros_like(value), tf.zeros_like(score)
        if case == 'center_nonfinite':
            value = tf.where(call_index == 0, tf.constant(float('nan'), D), value)
        if case in ('partial_design', 'insufficient'):
            prefix = 1 + 2 * directions if rank else 1
            is_design = (start + tf.range(extent, dtype=tf.int64) >= prefix) & (
                start + tf.range(extent, dtype=tf.int64) < prefix + samples)
            invalid = is_design & ((cloud[:, 0] > 0.) if case == 'partial_design' else tf.ones([extent], tf.bool))
            value = tf.where(invalid, tf.constant(float('nan'), D), value)
        return value, score

    def scalar(point):
        values, scores = compute(point[None], False)
        return values[0], scores[0]

    def batch(cloud):
        return compute(cloud, True)

    def reset():
        rows.assign(0)
        calls.assign(0)
        positions.assign(tf.zeros_like(positions))
        extents.assign(tf.zeros_like(extents))

    def record():
        return clean({'rows': int(rows), 'calls': int(calls), 'positions': positions.read_value(),
                      'extents': extents.read_value()})

    inputs = (center, scale, tf.constant(raw, D), tf.constant(offsets, D), tf.constant([726, 503]))
    return scalar, batch if batched else None, cfg, inputs, reset, record


def original(scalar, batch, cfg, inputs, monkeypatch):
    checkpoint, module = source('3582b4ac')

    class FrozenNumpy:
        random = SimpleNamespace(default_rng=lambda seed: SimpleNamespace(
            normal=lambda **kw: inputs[2].numpy().copy(),
            permutation=lambda count: _draw_kernel('permutation', (count,))(inputs[4]).numpy()))

        def __getattr__(self, name):
            return getattr(np, name)

    with monkeypatch.context() as patch:
        patch.setattr(module, 'np', FrozenNumpy())
        patch.setattr(module, '_sample_trust_ball', lambda *a, **kw: inputs[3].numpy().copy())
        result = module.fit_low_rank_spd_quadratic_geometry(scalar, inputs[0].numpy(),
            scale=inputs[1].numpy(), batched_value_and_score_fn=batch,
            config=module.LowRankSPDQuadraticGeometryConfig(**dataclasses.asdict(cfg)))
    return clean(result.payload(include_arrays=True)), checkpoint.hashes()


def comparison_payload(record):
    record = clean(record)
    # Byte hashes bind actual artifacts, not floating-point tolerance equality.
    record['diagnostics'].pop('artifact_hash', None)
    # The owner approved replacing the RNG; identical realized inputs above are
    # the numerical comparison authority, not a fabricated PCG64 identity.
    record['diagnostics'].pop('random_stream', None)
    record['diagnostics']['config'].pop('random_stream', None)
    # Preserve current public route names when comparing original records.
    # Only the historical scalar label is normalized. Raw saved records and
    # every numerical/discrete decision field remain unchanged.
    diagnostics = record['diagnostics']
    if diagnostics.get('design_evaluation_route') == 'scalar_value_and_score_loop':
        diagnostics['design_evaluation_route'] = 'tensorflow_scalar_row_loop'
    pilot = diagnostics.get('pilot', {})
    if pilot.get('evaluation_route') == 'scalar_value_and_score_loop':
        pilot['evaluation_route'] = 'tensorflow_scalar_row_loop'
    return record


def test_full_geometry_output_shapes(request):
    from bayesfilter.inference.quadratic_geometry_fit_tf import (
        make_geometry_fit_program,
    )

    scalar, _, cfg, _, _, _ = fixture(3, False)
    records = {}
    for jit in (False, True):
        program = make_geometry_fit_program(scalar, 3, 2, 12, 4, cfg,
            active_rows=True, minimum_train_rows=7, jit_compile=jit)
        outputs = program.get_concrete_function().structured_outputs
        records[str(jit)] = tf.nest.map_structure(
            lambda value: {'shape': str(value.shape), 'name': value.name},
            outputs)
        assert all(value.shape.is_fully_defined() for value in tf.nest.flatten(outputs))
    save(request, 'geometry-full-output-shapes.json', records)


@pytest.mark.parametrize('dimension', [1, 3, 5])
@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('case', ['gaussian', 'partial_directions', 'no_directions', 'partial_design',
    'insufficient', 'center_nonfinite', 'constant', 'no_holdout', 'nonquadratic', 'holdout_reject',
    'unconstrained', 'coupled'])
def test_full_geometry_original_records(dimension, batched, case, monkeypatch, request):
    scalar, batch, cfg, inputs, reset, record = fixture(dimension, batched, case)
    expected, hashes = original(scalar, batch, cfg, inputs, monkeypatch)
    expected_calls = record()
    candidates = {}
    for jit in (False, True):
        reset()
        program = make_geometry_program(scalar, dimension, cfg, batched_callback=batch, jit_compile=jit)
        raw = program(*inputs)
        result = geometry_result(raw, inputs[0], inputs[1], cfg, batched=batched)
        payload = clean(geometry_fit_payload(result))
        candidates[str(jit)] = {'payload': payload, 'calls': record(),
            'stage': int(raw['stage']), 'attempted_rows': int(raw['attempted_evaluation_count'])}
    save(request, f'geometry-full-{dimension}-{batched}-{case}.json', {'original': expected,
        'original_calls': expected_calls, 'source_sha256': hashes, 'inputs': inputs, 'candidate': candidates})
    for candidate in candidates.values():
        _equal_records(comparison_payload(candidate['payload']), comparison_payload(expected))
        _equal_records(candidate['calls'], expected_calls)
        assert candidate['attempted_rows'] == expected_calls['rows']


@pytest.mark.parametrize('batched', [False, True])
def test_unresolved_pilot_stops_before_design(batched, monkeypatch, request):
    from bayesfilter.inference import quadratic_geometry_full_tf as native

    scalar, batch, cfg, inputs, reset, record = fixture(3, batched, 'unresolved_pilot')
    results = {}
    for jit in (False, True):
        reset()
        raw = make_geometry_program(scalar, 3, cfg, batched_callback=batch, jit_compile=jit)(*inputs)
        result = geometry_result(raw, inputs[0], inputs[1], cfg, batched=batched)
        assert result.status == 'pilot_eigenbasis_ill_conditioned'
        assert result.precision is result.covariance is result.q_basis is None
        assert result.exact_evaluation_count == int(raw['attempted_evaluation_count']) == record()['rows'] == 3
        assert record()['calls'] == (2 if batched else 3)
        assert 'fit' not in result.diagnostics
        results[str(jit)] = geometry_fit_payload(result)
    reset()
    monkeypatch.setattr(native, 'prepare_geometry_inputs', lambda *args: inputs[2:])
    public = fit_low_rank_spd_quadratic_geometry(scalar, inputs[0], scale=inputs[1],
        batched_value_and_score_fn=batch, config=cfg)
    assert public.status == 'pilot_eigenbasis_ill_conditioned'
    assert public.precision is public.covariance is public.q_basis is None
    assert public.exact_evaluation_count == record()['rows'] == 3
    results['public'] = public.payload(include_arrays=True)
    save(request, f'geometry-full-pilot-rejection-{batched}.json', results)


@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('case', ['gaussian', 'partial_design', 'center_nonfinite'])
def test_public_geometry_uses_native_program(batched, case, monkeypatch, request):
    from bayesfilter.inference import quadratic_geometry_full_tf as native

    scalar, batch, cfg, inputs, reset, record = fixture(3, batched, case)
    inputs = (*inputs[:2], *prepare_geometry_inputs(3, cfg))
    expected, hashes = original(scalar, batch, cfg, inputs, monkeypatch)
    expected_calls = record()
    reset()
    selected = []
    factory = native.geometry_program

    def select(*args, **kwargs):
        result = factory(*args, **kwargs)
        selected.append(result)
        return result

    monkeypatch.setattr(native, 'geometry_program', select)
    actual = fit_low_rank_spd_quadratic_geometry(scalar, inputs[0], scale=inputs[1],
        batched_value_and_score_fn=batch, config=cfg)
    _equal_records(comparison_payload(actual.payload(include_arrays=True)), comparison_payload(expected))
    _equal_records(record(), expected_calls)
    if case != 'center_nonfinite':
        route = 'batched_value_and_score' if batched else 'tensorflow_scalar_row_loop'
        assert actual.diagnostics['design_evaluation_route'] == route
        assert actual.diagnostics['pilot']['evaluation_route'] == route
    assert len(selected) == 1
    assert selected[0].get_concrete_function().function_def.attr['_XlaMustCompile'].b
    assert selected[0].experimental_get_tracing_count() == 1
    save(request, f'geometry-full-public-{batched}-{case}.json',
        {'original': expected, 'actual': actual.payload(include_arrays=True), 'calls': record(),
         'original_source_sha256': hashes, 'public_program_jit_compile': True})


def test_complete_geometry_changing_operands_and_enclosure(monkeypatch, request):
    scalar, batch, cfg, inputs, reset, record = fixture(3, True, 'coupled')
    program = make_geometry_program(scalar, 3, cfg, batched_callback=batch)
    changed = (inputs[0] + .07, inputs[1] * 1.1,
        tf.concat((tf.zeros([1, 3], D), inputs[2][1:] * 1.2), 0), inputs[3] * .9, inputs[4] + 10)
    observations, hlos = [], []
    for args in (inputs, changed):
        reset()
        expected, hashes = original(scalar, batch, cfg, args, monkeypatch)
        expected_calls = record()
        reset()
        raw = program(*args)
        actual = geometry_fit_payload(geometry_result(raw, args[0], args[1], cfg, batched=True))
        _equal_records(comparison_payload(actual), comparison_payload(expected))
        _equal_records(record(), expected_calls)
        observations.append({'expected': expected, 'actual': actual, 'source_sha256': hashes, 'calls': record()})
        hlos.append(program.experimental_get_compiler_ir(*args)(stage='hlo'))
    assert program.experimental_get_tracing_count() == 1
    assert stable_pilot_hlo(hlos[0]) == stable_pilot_hlo(hlos[1])
    assert observations[0]['actual'] != observations[1]['actual']
    graph = program.get_concrete_function().graph.as_graph_def()
    assert not any(node.op in ('PyFunc', 'EagerPyFunc', 'PyFuncStateless')
        for nodes in (graph.node, *(fn.node_def for fn in graph.library.function)) for node in nodes)
    outer = tf.function(lambda *args: program(*args), input_signature=program.input_signature,
        autograph=False, jit_compile=True)
    reset()
    actual = geometry_fit_payload(geometry_result(outer(*changed), changed[0], changed[1], cfg, batched=True))
    _equal_records(comparison_payload(actual), comparison_payload(observations[1]['actual']))
    save(request, 'geometry-full-operands.json', {'observations': observations,
        'trace_count': 1, 'hlo_bytes': len(hlos[0].encode()), 'enclosing_xla': True})


@pytest.mark.parametrize('dimension', [1, 3])
def test_geometry_preparation_keeps_versioned_stream_order(dimension):
    cfg = LowRankSPDQuadraticGeometryConfig(sample_count=16, pilot_direction_count=6)
    raw, offsets, seed = prepare_geometry_inputs(dimension, cfg)
    stream = GeometryTensorStream(cfg.seed)
    if dimension > 1:
        np.testing.assert_array_equal(raw, stream.normal(size=(6, dimension)))
    else:
        assert raw.shape == (0, dimension)
    np.testing.assert_array_equal(offsets, stream.ball(16, dimension, radius=cfg.trust_radius))
    np.testing.assert_array_equal(seed, stream._next_seed('permutation'))


def test_restricted_fit_count_domain_preserves_original(monkeypatch, request):
    from bayesfilter.inference.quadratic_geometry_fit_tf import (
        make_geometry_fit_program,
    )
    from tests.test_filter_repair_geometry_active_rows import active_fixture
    from tests.test_filter_repair_geometry_fit import materialize, original_result

    target, cfg, _, _, calls = active_fixture(3, 'gaussian', 12, 4, capacity=12)
    program = make_geometry_fit_program(target, 3, 2, 12, 4, cfg,
        active_rows=True, minimum_train_rows=7)
    records = []
    for finite_count in range(7, 17):
        holdout = min(finite_count // 4, finite_count - 7)
        training = finite_count - holdout
        _, local, compact, padded, _ = active_fixture(3, 'gaussian', training, holdout, capacity=12)
        calls.assign(0)
        expected, _, hashes = original_result(target, local, compact, monkeypatch)
        expected_calls = int(calls)
        calls.assign(compact[-1])
        actual = materialize(program(*padded), local, compact)
        _equal_records(actual, expected)
        assert int(calls) == expected_calls
        records.append({'finite_count': finite_count, 'original': expected,
                        'actual': actual, 'source_sha256': hashes})
    calls.assign(0)
    rejected = program(*padded[:-2], tf.constant(6), tf.constant(4))
    assert not bool(rejected['fit']['finite']) and int(calls) == 0
    save(request, 'geometry-full-restricted-counts.json', records)


@pytest.mark.parametrize('batched', [False, True])
def test_capacity_below_finite_requirement_never_fits(batched, monkeypatch, request):
    scalar, batch, cfg, inputs, reset, record = fixture(3, batched, samples=2)
    expected, hashes = original(scalar, batch, cfg, inputs, monkeypatch)
    expected_calls = record()
    reset()
    raw = make_geometry_program(scalar, 3, cfg, batched_callback=batch)(*inputs)
    actual = geometry_fit_payload(geometry_result(raw, inputs[0], inputs[1], cfg, batched=batched))
    _equal_records(comparison_payload(actual), comparison_payload(expected))
    _equal_records(record(), expected_calls)
    assert actual['status'] == 'insufficient_finite_samples'
    assert raw['fit_result'] == {}
    save(request, f'geometry-full-impossible-count-{batched}.json',
        {'original': expected, 'actual': actual, 'source_sha256': hashes, 'calls': record()})
