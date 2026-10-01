"""Diagnostic fresh-process cost of structured preparation and padded fitting."""

import hashlib
import inspect
import json
import subprocess
import time
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference import factor_correlation_geometry as factor
from bayesfilter.inference import sequential_map_covariance as sequential
from bayesfilter.inference.sequential_structured_fit_tf import (
    structured_fit_data_program,
)
from bayesfilter.inference.sequential_structured_preparation_tf import (
    structured_data_program,
)
from tests.test_filter_repair_factor_capacity import _memory
from tests.test_filter_repair_structured_fit import frozen as _frozen

D = tf.float64


def _inputs(dimension, capacity):
    center = tf.linspace(tf.constant(-.2, D), tf.constant(.3, D), dimension)
    scale = tf.linspace(tf.constant(.8, D), tf.constant(1.2, D), dimension)
    scales = tf.constant([.8, 1.2, 1.4] if dimension == 3 else [.8, 1.1, .9, 1.2, .7], D)
    loadings = tf.constant([[.2], [-.1], [.3]] if dimension == 3 else
        [[.3, 0.], [.12, .25], [-.2, .1], [.15, -.1], [.08, .2]], D)
    covariance = scales[:, None] * (tf.linalg.diag(1. - tf.reduce_sum(loadings ** 2, axis=1))
        + tf.matmul(loadings, loadings, transpose_b=True)) * scales[None, :]
    precision = tf.linalg.inv(covariance)

    def batched(rows):
        scores = -(rows @ precision)
        return .5 * tf.reduce_sum(rows * scores, axis=1), scores

    def scalar(row):
        values, scores = batched(row[None, :])
        return values[0], scores[0]

    offsets = tf.concat([tf.constant([[.07] + [0.] * (dimension - 1)], D),
        tf.zeros([capacity - 1, dimension], D)], 0)
    search = center[None, :] + offsets * scale[None, :]
    return scalar, batched, (center, scalar(center)[1], scale, tf.constant(.3, D),
        tf.constant([2026, 715]), search, batched(search)[1])


@pytest.mark.parametrize('dimension,capacity', [(3, 4), (5, 32)])
@pytest.mark.parametrize('arm', ['before', 'after', 'graph', 'xla'])
def test_structured_preparation_fit_memory(dimension, capacity, arm, request):
    module = _frozen.__wrapped__() if arm == 'before' else sequential
    baseline_sources = {path: hashlib.sha256(subprocess.check_output(
        ['git', 'show', 'f06fd505:' + path])).hexdigest() for path in
        ('bayesfilter/inference/factor_correlation_geometry.py',
         'bayesfilter/inference/sequential_map_covariance.py')}
    stages = {'before': _memory()}
    scalar, batched, arguments = _inputs(dimension, capacity)
    hashes = [hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest() for value in arguments]
    factors = 1 if dimension == 3 else 2
    config = sequential.SequentialMapCovarianceConfig()
    factor_config = factor.FactorCorrelationGeometryConfig(factor_count=factors)
    jit = arm != 'graph'
    if arm in ('graph', 'xla'):
        prepare = structured_data_program(scalar, batched, dimension, 4 * dimension, capacity, True, jit_compile=jit)

        @tf.function(input_signature=prepare.input_signature, jit_compile=jit, autograph=False)
        def complete(*args):
            data = prepare.python_function(*args)
            fit = structured_fit_data_program(dimension, 2 * dimension + capacity, 2 * dimension,
                factor_config, jit_compile=jit).python_function(data['center_score_z'], data['training_offsets_z'],
                data['training_scores_z'], data['holdout_offsets_z'], data['holdout_scores_z'],
                data['training_weights'], data['active_training_rows'])
            return data, fit

    def execute():
        if arm in ('before', 'after'):
            center, score, scale, radius, seed, search, search_scores = arguments
            data, evaluations = module._structured_factor_fit_data(scalar, center, score, scale,
                dimension=dimension, fresh_sample_count=4 * dimension, radius=float(radius),
                seed=tuple(seed.numpy().tolist()), search_theta=search, search_scores=search_scores,
                reuse_search_scores=True, evaluations=7, batched_value_and_score_fn=batched)
            result = module._fit_factor_from_data(data, factor_count=factors, config=config)
        else:
            data, computed = complete(*arguments)
            assert int(computed['input_status']) == 0
            assert int(computed['fit']['invalid_covariance_evaluations']) == 0
            result = dict(factor._factor_result_from_computed(computed['fit'], factor_config,
                dimension, int(data['active_training_rows']), 2 * dimension, jit).payload())
            winner = int(data['best_index']) >= 0
            result.update(status=result['status'], rank=result['parameter_count'],
                projected_precision_z=result['precision_z'], projection_relative_frobenius=0.,
                fresh_training_count=int(data['fresh_training_count']),
                fresh_holdout_count=int(data['fresh_holdout_count']),
                reused_training_count=int(data['reused_training_count']),
                best_exact_value=float(data['best_value']) if winner else None,
                best_exact_position=data['best_position'].numpy().tolist() if winner else None,
                best_exact_score=data['best_score'].numpy().tolist() if winner else None,
                best_exact_source='structured_fit_cloud' if winner else None)
            evaluations = 7 + int(data['unique_fresh_evaluations'])
        return {'evaluations': evaluations, 'fit': result}

    stages['built'] = _memory()
    samples = []
    first = None
    for index in range(21):
        tf.config.experimental.reset_memory_stats('GPU:0')
        started = time.perf_counter()
        result = execute()
        elapsed = time.perf_counter() - started
        assert result['fit']['status'] == 'usable'
        assert result['fit']['covariance_z'] is not None
        samples.append({'seconds': elapsed, 'memory': _memory()})
        if first is None:
            first = result
        else:
            assert result == first
    stages['measured'] = _memory()
    if arm in ('graph', 'xla'):
        trace_count = complete.experimental_get_tracing_count()
        hlo = complete.experimental_get_compiler_ir(*arguments)(stage='hlo') if jit else ''
        graph = complete.get_concrete_function().graph.as_graph_def()
        nodes = len(graph.node) + sum(len(function.node_def) for function in graph.library.function)
        assert trace_count in (1, 2)  # First call may create the guard resource.
    else:
        trace_count, nodes, hlo = None, None, ''
    sources = {str(Path(item.__file__).relative_to(Path.cwd())):
        hashlib.sha256(Path(item.__file__).read_bytes()).hexdigest() for item in
        (factor, sequential, inspect.getmodule(structured_fit_data_program),
         inspect.getmodule(structured_data_program))}
    report = {'role': 'descriptive_structured_preparation_fit_checkpoint_cost', 'checkpoint': 'f06fd505',
        'arm': arm, 'dimension': dimension, 'capacity': capacity, 'reused_rows': 1,
        'fresh_rows': 4 * dimension, 'factor_count': factors, 'max_iterations': 200,
        'jit_compile': jit, 'non_jit_role': None if jit else 'explicit_graph_reference_exception',
        'input_sha256': hashes, 'source_sha256': sources, 'baseline_source_sha256': baseline_sources, 'samples': samples, 'result': first,
        'stages': stages, 'trace_count': trace_count, 'graph_nodes': nodes, 'hlo_bytes': len(hlo.encode()),
        'timing_scope': 'complete_preparation_fit_and_all_public_fields_materialized',
        'nonclaims': ['One fresh process per arm; no statistical timing ranking or terminal repeats.',
            'Outer sequential controller remains outside this numerical boundary.']}
    directory = Path(request.config.getoption('xmlpath')).parent
    with (directory / 'structured-memory.json').open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    if hlo:
        with (directory / 'structured-memory-hlo.txt').open('x') as handle:
            handle.write(hlo)


def test_structured_graph_xla_localization(request):
    """Explanatory crossed-input fits; preserves failed fields for diagnosis."""
    from tests.test_filter_repair_initializer_rounding import _record_differences
    from tests.test_filter_repair_structured_fit import _prepared_arguments

    dimension, capacity = 5, 32
    scalar, batched, arguments = _inputs(dimension, capacity)
    config = factor.FactorCorrelationGeometryConfig(factor_count=2)
    prepared = {mode: structured_data_program(scalar, batched, dimension,
        4 * dimension, capacity, True, jit_compile=jit)(*arguments)
        for mode, jit in [('graph', False), ('xla', True)]}

    def materialize(record):
        return tf.nest.map_structure(lambda value: value.numpy().tolist(), record)

    records = {}
    for data_mode, data in prepared.items():
        for fit_mode, jit in [('graph', False), ('xla', True)]:
            computed = structured_fit_data_program(dimension, 2 * dimension + capacity,
                2 * dimension, config, jit_compile=jit)(*_prepared_arguments(data))
            assert int(computed['input_status']) == 0
            record = factor._factor_result_from_computed(computed['fit'], config,
                dimension, int(data['active_training_rows']), 2 * dimension, jit).payload()
            records[data_mode + '_data_' + fit_mode + '_fit'] = dict(record)
    data = prepared['xla']
    count = int(data['active_training_rows'])
    for mode, jit in [('graph', False), ('xla', True)]:
        result = factor.fit_factor_correlation_score_geometry(data['center_score_z'],
            data['training_offsets_z'][:count], data['training_scores_z'][:count],
            data['holdout_offsets_z'], data['holdout_scores_z'],
            training_weights=data['training_weights'][:count], config=config, jit_compile=jit)
        records['compact_' + mode] = dict(result.payload())
    reference = records['xla_data_xla_fit']
    report = {'role': 'explanatory_crossed_input_localization',
        'prepared': {mode: materialize(data) for mode, data in prepared.items()},
        'records': records,
        'record_differences_from_xla': {name: _record_differences(record, reference)
            for name, record in records.items()},
        'nonclaims': ['Instrumented localization; not a substitute for the complete cost comparison.']}
    directory = Path(request.config.getoption('xmlpath')).parent
    with (directory / 'structured-graph-xla-localization.json').open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')


def test_inherited_structured_fitter_modes(request):
    """Distinguish inherited compiler rounding from this checkpoint's repair."""
    from tests.test_filter_repair_fixed_stability import _compare
    from tests.test_filter_repair_initializer_rounding import _record_differences

    frozen = _frozen.__wrapped__()
    scalar, batched, arguments = _inputs(5, 32)
    data = structured_data_program(scalar, batched, 5, 20, 32, True)(*arguments)
    count = int(data['active_training_rows'])
    records = {}
    for mode, jit in [('graph', False), ('xla', True)]:
        for label, module in [('before', frozen), ('after', factor)]:
            result = module.fit_factor_correlation_score_geometry(data['center_score_z'],
                data['training_offsets_z'][:count], data['training_scores_z'][:count],
                data['holdout_offsets_z'], data['holdout_scores_z'],
                training_weights=data['training_weights'][:count],
                config=module.FactorCorrelationGeometryConfig(factor_count=2), jit_compile=jit)
            records[label + '_' + mode] = dict(result.payload())
        _compare(records['after_' + mode], records['before_' + mode])
    report = {'role': 'frozen_checkpoint_compiler_rounding_localization', 'records': records,
        'graph_xla_differences': {label: _record_differences(records[label + '_graph'], records[label + '_xla'])
            for label in ('before', 'after')},
        'baseline': 'f06fd505',
        'nonclaims': ['Same-mode parity does not waive the failed graph/XLA gate.']}
    directory = Path(request.config.getoption('xmlpath')).parent
    with (directory / 'structured-inherited-modes.json').open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')


@pytest.mark.parametrize('arm', ['before', 'after'])
def test_structured_changing_eligibility_memory(arm, request):
    """Measure first use and reuse of six active sizes in separate processes."""
    module = _frozen.__wrapped__() if arm == 'before' else sequential
    stages = {'before': _memory()}
    scalar, batched, original = _inputs(5, 32)
    center, score, scale, radius, seed, _, _ = original
    offset = tf.constant([[.07, 0., 0., 0., 0.]], D)
    config = sequential.SequentialMapCovarianceConfig()
    schedule = [1, 0, 3, 8, 16, 32] * 2
    samples, records = [], {}
    stages['built'] = _memory()
    for count in schedule:
        offsets = tf.concat([tf.repeat(offset, count, axis=0), tf.zeros([32 - count, 5], D)], 0)
        search = center[None, :] + offsets * scale[None, :]
        search_scores = batched(search)[1]
        hashes = [hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest()
            for value in (center, score, scale, radius, seed, search, search_scores)]
        tf.config.experimental.reset_memory_stats('GPU:0')
        start = time.perf_counter()
        data, evaluations = module._structured_factor_fit_data(scalar, center, score, scale,
            dimension=5, fresh_sample_count=20, radius=float(radius), seed=tuple(seed.numpy().tolist()),
            search_theta=search, search_scores=search_scores, reuse_search_scores=True,
            evaluations=7, batched_value_and_score_fn=batched)
        result = module._fit_factor_from_data(data, factor_count=2, config=config)
        elapsed = time.perf_counter() - start
        record = {'evaluations': evaluations, 'fit': result}
        assert result['reused_training_count'] == count
        if count in records:
            assert record == records[count]
        records[count] = record
        samples.append({'reused_rows': count, 'seconds': elapsed, 'memory': _memory(),
            'input_sha256': hashes, 'result': record})
    report = {'role': 'descriptive_changing_eligibility_cost', 'arm': arm, 'checkpoint': 'f06fd505',
        'dimension': 5, 'capacity': 32, 'max_iterations': 200, 'schedule': schedule,
        'stages': stages, 'samples': samples,
        'source_sha256': {str(Path(item.__file__).relative_to(Path.cwd())):
            hashlib.sha256(Path(item.__file__).read_bytes()).hexdigest() for item in
            (factor, sequential, inspect.getmodule(structured_fit_data_program), inspect.getmodule(structured_data_program))},
        'nonclaims': ['One process per arm; repeated eligible rows isolate active shapes, not scientific target coverage.',
            'Does not waive any failed parity gate or admit unbounded capacity.']}
    directory = Path(request.config.getoption('xmlpath')).parent
    with (directory / 'structured-changing-eligibility.json').open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
