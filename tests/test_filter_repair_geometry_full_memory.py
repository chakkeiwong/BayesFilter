"""Fresh-process descriptive complete geometry costs on identical prepared inputs."""

import dataclasses
import hashlib
import time
from types import SimpleNamespace

import pytest
import tensorflow as tf

from bayesfilter.inference.quadratic_geometry import LowRankSPDQuadraticGeometryConfig
from bayesfilter.inference.quadratic_geometry_full_report import geometry_result
from bayesfilter.inference.quadratic_geometry_full_tf import (
    make_geometry_program,
    prepare_geometry_inputs,
)
from bayesfilter.ops.geometry_random_tf import _draw_kernel
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_gap_diagnostics import memory_snapshot
from tests.test_filter_repair_geometry_control import clean, save
from tests.test_filter_repair_geometry_full import comparison_payload, original
from tests.test_filter_repair_quadratic_batches import _equal_records

D = tf.float64


@pytest.mark.parametrize('capacity', [24, 120])
@pytest.mark.parametrize('arm', ['prior', 'prior_refined', 'graph', 'xla'])
def test_full_geometry_costs(arm, capacity, monkeypatch, request):
    cfg = LowRankSPDQuadraticGeometryConfig(rank=2, sample_count=capacity,
        min_samples_per_parameter=1, pilot_direction_count=6 if capacity == 24 else 64,
        constrain_center_refinement_to_trust_region=True)
    inputs = (tf.constant([.01, .02, .03], D), tf.constant([.8, .9, 1.], D),
              *prepare_geometry_inputs(3, cfg))

    def batch(points):
        delta = points - .13
        scores = -delta * tf.constant([2., 3., 4.], D) - .2 * delta ** 3
        values = -.5 * tf.reduce_sum(delta ** 2 * tf.constant([2., 3., 4.], D), axis=1)
        return values - .05 * tf.reduce_sum(delta ** 4, axis=1), scores

    def scalar(point):
        values, scores = batch(point[None])
        return values[0], scores[0]

    checkpoint = FrozenCheckpoint('91928762e', 'full_geometry_cost_prior')
    prior = checkpoint.load('bayesfilter.inference.quadratic_geometry')
    prior_config = prior.LowRankSPDQuadraticGeometryConfig(**dataclasses.asdict(cfg))
    frozen = SimpleNamespace(normal=lambda **kw: inputs[2], ball=lambda *a, **kw: inputs[3],
        permutation=lambda count: _draw_kernel('permutation', (count,))(inputs[4]))
    monkeypatch.setattr(prior, 'GeometryTensorStream', lambda seed: frozen)
    if arm == 'prior_refined':
        from bayesfilter.inference.mass_matrix_tf import eigenpair_program

        original_kernel = prior._trust_region_kernel

        def repaired_eigen_kernel(matrix, vector, radius):
            return original_kernel(matrix, vector, radius, eigenpairs=eigenpair_program(3))

        monkeypatch.setattr(prior, '_trust_region_kernel', repaired_eigen_kernel)
        original_pilot = prior._pilot_sketch_kernel

        def repaired_pilot(directions, plus, minus, scale, step):
            return original_pilot(directions, plus, minus, scale, step, eigenpairs=eigenpair_program(3))

        monkeypatch.setattr(prior, '_pilot_sketch_kernel', repaired_pilot)
    gpu = bool(tf.config.list_logical_devices('GPU'))
    stages = {'prepared': memory_snapshot(gpu)}
    started = time.perf_counter()
    program = None if arm in ('prior', 'prior_refined') else make_geometry_program(scalar, 3, cfg,
        batched_callback=batch, jit_compile=arm == 'xla')
    build_seconds = time.perf_counter() - started
    stages['built'] = memory_snapshot(gpu)
    started = time.perf_counter()
    if program is not None:
        program.get_concrete_function()
    trace_seconds = time.perf_counter() - started
    stages['traced'] = memory_snapshot(gpu)

    def execute():
        if gpu:
            tf.config.experimental.reset_memory_stats('GPU:0')
        begin = time.perf_counter()
        if program is None:
            result = prior.fit_low_rank_spd_quadratic_geometry(scalar, inputs[0], scale=inputs[1],
                batched_value_and_score_fn=batch, config=prior_config)
            native = None
        else:
            raw = program(*inputs)
            raw['stage'].numpy()
            native = time.perf_counter() - begin
            result = geometry_result(raw, inputs[0], inputs[1], cfg, batched=True)
        payload = clean(result.payload(include_arrays=True))
        elapsed = time.perf_counter() - begin
        return payload, {'seconds': elapsed, 'native_seconds': native,
            'report_seconds': None if native is None else elapsed - native,
            'memory': memory_snapshot(gpu)}

    first, cold = execute()
    stages['cold'] = memory_snapshot(gpu)
    samples = []
    for _ in range(20):
        payload, sample = execute()
        assert payload == first
        samples.append(sample)
    stages['warm'] = memory_snapshot(gpu)
    # IR and numerical authority are intentionally inspected after all costs.
    graph_nodes, graph_bytes, hlo_bytes, traces = None, None, None, None
    if program is not None:
        graph = program.get_concrete_function().graph.as_graph_def()
        graph_nodes = len(graph.node) + sum(len(fn.node_def) for fn in graph.library.function)
        graph_bytes = graph.ByteSize()
        traces = program.experimental_get_tracing_count()
        assert traces == 1
        if arm == 'xla':
            hlo_bytes = len(program.experimental_get_compiler_ir(*inputs)(stage='hlo').encode())
    expected, hashes = original(scalar, batch, cfg, inputs, monkeypatch)
    save(request, 'geometry-full-memory.json', {'arm': arm, 'capacity': capacity,
        'direction_capacity': cfg.pilot_direction_count, 'gpu': gpu, 'jit_compile': arm == 'xla',
        'numerical_authority': '3582b4ac', 'mechanism_baseline': '91928762e',
        'prior_comparator_patch': 'shared refined eigensystem in trust-region and pilot-sketch kernels only' if arm == 'prior_refined' else None,
        'prior_source_sha256': checkpoint.hashes(), 'original_source_sha256': hashes,
        'input_sha256': [hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest() for value in inputs],
        'build_seconds': build_seconds, 'trace_seconds': trace_seconds, 'stages': stages,
        'cold': cold, 'samples': samples, 'payload': first, 'original_payload': expected,
        'graph_nodes': graph_nodes, 'graph_bytes': graph_bytes, 'hlo_bytes': hlo_bytes, 'trace_count': traces,
        'scope': 'Complete center/pilot/design/partition/fit/refinement/replay and public payload; prepared random inputs excluded equally.',
        'execution_role': 'mixed_host_and_compiled' if program is None else ('explicit_graph_reference' if arm == 'graph' else 'native_xla'),
        'non_jit_role': 'explicit_graph_reference' if arm == 'graph' else None,
        'nonclaims': ['Single-process descriptive costs; no timing rank, leak-freedom, terminal acceptance or whole iterative qualification.']})
    _equal_records(comparison_payload(first), comparison_payload(expected))


@pytest.mark.parametrize('capacity', [24, 120])
def test_prior_geometry_spectral_attribution(capacity, monkeypatch, request):
    """Distinguish old raw-eigen error from enclosing-program arithmetic."""
    from bayesfilter.inference.mass_matrix_tf import eigenpair_program

    cfg = LowRankSPDQuadraticGeometryConfig(rank=2, sample_count=capacity,
        min_samples_per_parameter=1, pilot_direction_count=6 if capacity == 24 else 64,
        constrain_center_refinement_to_trust_region=True)
    inputs = (tf.constant([.01, .02, .03], D), tf.constant([.8, .9, 1.], D),
              *prepare_geometry_inputs(3, cfg))

    def batch(points):
        delta = points - .13
        scores = -delta * tf.constant([2., 3., 4.], D) - .2 * delta ** 3
        values = -.5 * tf.reduce_sum(delta ** 2 * tf.constant([2., 3., 4.], D), axis=1)
        return values - .05 * tf.reduce_sum(delta ** 4, axis=1), scores

    def scalar(point):
        values, scores = batch(point[None])
        return values[0], scores[0]

    expected, hashes = original(scalar, batch, cfg, inputs, monkeypatch)
    checkpoint = FrozenCheckpoint('91928762e', 'full_geometry_prior_eigen_attribution')
    prior = checkpoint.load('bayesfilter.inference.quadratic_geometry')
    prior_config = prior.LowRankSPDQuadraticGeometryConfig(**dataclasses.asdict(cfg))
    frozen = SimpleNamespace(normal=lambda **kw: inputs[2], ball=lambda *a, **kw: inputs[3],
        permutation=lambda count: _draw_kernel('permutation', (count,))(inputs[4]))
    monkeypatch.setattr(prior, 'GeometryTensorStream', lambda seed: frozen)

    def prior_call():
        return prior.fit_low_rank_spd_quadratic_geometry(scalar, inputs[0], scale=inputs[1],
            batched_value_and_score_fn=batch, config=prior_config).payload(include_arrays=True)

    records = {'original': expected, 'prior': prior_call()}
    old_kernel = prior._trust_region_kernel
    eigenpairs = eigenpair_program(3)

    def refined_kernel(matrix, vector, radius):
        return old_kernel(matrix, vector, radius, eigenpairs=eigenpairs)

    monkeypatch.setattr(prior, '_trust_region_kernel', refined_kernel)
    records['prior_refined_eigen_only'] = prior_call()
    old_pilot = prior._pilot_sketch_kernel

    def refined_pilot(directions, plus, minus, scale, step):
        return old_pilot(directions, plus, minus, scale, step, eigenpairs=eigenpairs)

    monkeypatch.setattr(prior, '_pilot_sketch_kernel', refined_pilot)
    records['prior_both_eigensystems_refined'] = prior_call()
    for jit in (False, True):
        program = make_geometry_program(scalar, 3, cfg, batched_callback=batch, jit_compile=jit)
        records[f'candidate_{jit}'] = geometry_result(program(*inputs), *inputs[:2], cfg,
            batched=True).payload(include_arrays=True)
    save(request, f'geometry-full-prior-eigen-attribution-{capacity}.json', {
        'records': records, 'original_source_sha256': hashes,
        'prior_source_sha256': checkpoint.hashes(),
        'diagnostic_only_patch': 'Prior trust-region then pilot-sketch eigenpairs callable uses shared refined eigenpair_program',
        'input_sha256': [hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest() for value in inputs]})
    with pytest.raises(AssertionError):
        _equal_records(comparison_payload(records['prior']), comparison_payload(expected))
    if capacity == 24:
        _equal_records(comparison_payload(records['prior_refined_eigen_only']), comparison_payload(expected))
    else:
        with pytest.raises(AssertionError):
            _equal_records(comparison_payload(records['prior_refined_eigen_only']), comparison_payload(expected))
    for arm in ('prior_both_eigensystems_refined', 'candidate_False', 'candidate_True'):
        _equal_records(comparison_payload(records[arm]), comparison_payload(expected))
