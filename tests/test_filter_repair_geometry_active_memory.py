"""Fresh-process descriptive costs of compact versus active-count fit programs."""

import hashlib
import time

import pytest
import tensorflow as tf

from bayesfilter.inference import quadratic_geometry_fit_report as report
from bayesfilter.inference.quadratic_geometry_fit_tf import make_geometry_fit_program
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.filter_repair_geometry_fit_reference import prefix_context, result_suffix
from tests.test_filter_repair_gap_diagnostics import memory_snapshot
from tests.test_filter_repair_geometry_active_rows import active_fixture
from tests.test_filter_repair_geometry_control import clean, save, stable_hlo
from tests.test_filter_repair_geometry_fit import original_result
from tests.test_filter_repair_quadratic_batches import _equal_records


@pytest.mark.parametrize('capacity', [16, 24])
@pytest.mark.parametrize('arm', ['compact', 'graph', 'xla'])
def test_active_fit_costs(arm, capacity, monkeypatch, request):
    target, cfg, compact, padded, _ = active_fixture(3, 'nonquadratic', 12, 2,
        capacity=capacity, counter=False)
    _, changed_cfg, changed, changed_padded, _ = active_fixture(3, 'nonquadratic', 13, 2,
        capacity=capacity, counter=False, shift=.07)
    contexts = (prefix_context(target, cfg, compact), prefix_context(target, changed_cfg, changed))
    checkpoint = FrozenCheckpoint('ca920bac5', 'active_fit_cost_baseline')
    prior = checkpoint.load('bayesfilter.inference.quadratic_geometry_fit_tf')
    prior_report = checkpoint.load('bayesfilter.inference.quadratic_geometry_fit_report')
    gpu = bool(tf.config.list_logical_devices('GPU'))
    stages = {'prepared': memory_snapshot(gpu)}
    start = time.perf_counter()
    if arm == 'compact':
        program = prior.make_geometry_fit_program(target, 3, 2, 12, 2, cfg)
        inputs, adapter = compact, prior_report
    else:
        program = make_geometry_fit_program(target, 3, 2, capacity, 4, cfg,
            active_rows=True, jit_compile=arm == 'xla')
        inputs, adapter = padded, report
    build_seconds = time.perf_counter() - start
    stages['built'] = memory_snapshot(gpu)
    start = time.perf_counter()
    program.get_concrete_function()
    trace_seconds = time.perf_counter() - start
    stages['traced'] = memory_snapshot(gpu)

    def execute(function, values, compact_values, config, context):
        if gpu:
            tf.config.experimental.reset_memory_stats('GPU:0')
        start = time.perf_counter()
        raw = function(*values)
        raw['status'].numpy()
        native_seconds = time.perf_counter() - start
        result = adapter.geometry_fit_result(raw, config, compact_values[0], compact_values[1],
            context[2], holdout_rows=compact_values[6].shape[0])
        fields = clean(result_suffix(result))
        payload = clean(adapter.geometry_fit_payload(result))
        seconds = time.perf_counter() - start
        return fields, payload, {'seconds': seconds, 'native_seconds': native_seconds,
            'report_seconds': seconds - native_seconds, 'memory': memory_snapshot(gpu)}

    first, first_payload, cold = execute(program, inputs, compact, cfg, contexts[0])
    stages['first_execution'] = memory_snapshot(gpu)
    samples = []
    for _ in range(20):
        fields, payload, sample = execute(program, inputs, compact, cfg, contexts[0])
        assert fields == first and payload == first_payload
        samples.append(sample)
    stages['warm'] = memory_snapshot(gpu)
    start = time.perf_counter()
    changed_program = (prior.make_geometry_fit_program(target, 3, 2, 13, 2, changed_cfg)
        if arm == 'compact' else program)
    changed_build_seconds = time.perf_counter() - start
    changed_inputs = changed if arm == 'compact' else changed_padded
    second, second_payload, changed_cost = execute(changed_program, changed_inputs, changed,
        changed_cfg, contexts[1])
    stages['changed_count'] = memory_snapshot(gpu)
    changed_samples = []
    for index in range(20):
        function, values, compact_values, config, context, expected, expected_payload = (
            (program, inputs, compact, cfg, contexts[0], first, first_payload) if index % 2 == 0 else
            (changed_program, changed_inputs, changed, changed_cfg, contexts[1], second, second_payload))
        fields, payload, sample = execute(function, values, compact_values, config, context)
        assert fields == expected and payload == expected_payload
        changed_samples.append(sample)
    stages['changing_count_reuse'] = memory_snapshot(gpu)
    # IR inspection and independent-reference evaluation occur after all costs.
    graph = program.get_concrete_function().graph.as_graph_def()
    nodes = len(graph.node) + sum(len(fn.node_def) for fn in graph.library.function)
    hlo = changed_hlo = ''
    if arm != 'graph':
        hlo = program.experimental_get_compiler_ir(*inputs)(stage='hlo')
        changed_hlo = changed_program.experimental_get_compiler_ir(*changed_inputs)(stage='hlo')
        if arm == 'xla':
            assert stable_hlo(hlo) == stable_hlo(changed_hlo)
    assert program.experimental_get_tracing_count() == changed_program.experimental_get_tracing_count() == 1
    expected, original_payload, original_hashes = original_result(target, cfg, compact, monkeypatch)
    expected_changed, original_changed_payload, _ = original_result(target, changed_cfg, changed, monkeypatch)
    save(request, 'geometry-active-memory.json', {'role': 'descriptive_active_count_fit_cost',
        'baseline': 'ca920bac5_compact_xla', 'numerical_authority': '3582b4ac', 'arm': arm,
        'dimension': 3, 'capacity': capacity, 'counts': [[12, 2], [13, 2]], 'gpu': gpu,
        'jit_compile': arm != 'graph', 'baseline_source_sha256': checkpoint.hashes(),
        'original_source_sha256': original_hashes,
        'compact_input_sha256': [hashlib.sha256(tf.io.serialize_tensor(v).numpy()).hexdigest() for v in compact],
        'changed_compact_input_sha256': [hashlib.sha256(tf.io.serialize_tensor(v).numpy()).hexdigest() for v in changed],
        'build_seconds': build_seconds, 'trace_seconds': trace_seconds, 'stages': stages,
        'cold': cold, 'samples': samples, 'changed_build_seconds': changed_build_seconds,
        'changed_cost': changed_cost, 'changed_samples': changed_samples,
        'result': first, 'changed_result': second, 'original_result': expected,
        'original_changed_result': expected_changed, 'payload': first_payload,
        'changed_payload': second_payload, 'original_payload': original_payload,
        'original_changed_payload': original_changed_payload, 'trace_count': 1,
        'program_instances': 2 if arm == 'compact' else 1, 'graph_nodes': nodes,
        'hlo_bytes': len(hlo.encode()), 'changed_hlo_bytes': len(changed_hlo.encode()),
        'non_jit_role': 'explicit_reference_exception' if arm == 'graph' else None,
        'timing_scope': 'Fit through full result/hash/payload; identical evaluated prefix excluded.',
        'nonclaims': ['Single-process descriptive component costs; no ranking, leak-freedom or public-controller qualification.']})
    _equal_records(first, expected)
    _equal_records(second, expected_changed)
    for actual, original in ((first_payload, original_payload), (second_payload, original_changed_payload)):
        actual['diagnostics'].pop('artifact_hash')
        original['diagnostics'].pop('artifact_hash')
        _equal_records(actual, original)
