"""Diagnostic fresh-process cost of exact old/current attempt-block excerpts."""

import hashlib
import inspect
import json
import re
import textwrap
import time
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference import factor_correlation_geometry as factor
from bayesfilter.inference import sequential_attempts_tf as native
from bayesfilter.inference import sequential_factor_attempt_tf as provider
from bayesfilter.inference import sequential_map_covariance as current
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_factor_capacity import _memory
from tests.test_filter_repair_structured_memory import _inputs

D = tf.float64


def _excerpt(module, source, *, old):
    start = source.index('        proposal_rows: list[Mapping[str, Any]] = []' if old else
        '        first_usable = fit["status"] == "usable"')
    stop = source.index('        _emit_progress(', start)
    block = textwrap.dedent(source[start:stop])
    wrapper = ('def run(value_and_score_fn, dimension, cfg, center, center_value, center_score, '
        'scale_tf, radius, stalled, fit, structured_data):\n'
        '    history, row_diag, evaluations = [], {}, 0\n'
        + textwrap.indent(block, '    ')
        + '    return {"history": history, "evaluations": evaluations, "center": center, '
        '"center_value": center_value, "center_score": center_score, "stalled": stalled, "radius": radius}\n')
    namespace = dict(vars(module))
    exec(compile(wrapper, 'diagnostic_attempts_exact_excerpt', 'exec'), namespace)  # noqa: S102 - verbatim runtime slice
    return namespace['run'], wrapper


@pytest.mark.parametrize('dimension,capacity', [(3, 4), (5, 32)])
@pytest.mark.parametrize('arm', ['before', 'graph', 'xla'])
def test_attempts_memory(dimension, capacity, arm, request):
    _measure_attempts(dimension, capacity, arm, request, warm_calls=20)


@pytest.mark.parametrize('capacity', [4, 32])
@pytest.mark.parametrize('arm', ['before', 'xla'])
def test_attempts_enclosure_capacity(arm, capacity, request):
    _measure_attempts(5, capacity, arm, request, warm_calls=3)


def _measure_attempts(dimension, capacity, arm, request, warm_calls):
    checkpoint = FrozenCheckpoint('93c8e419', 'attempts_cost')
    frozen = checkpoint.load('bayesfilter.inference.sequential_map_covariance')
    setup = frozen if arm == 'before' else current
    stages = {'before_preparation': _memory()}
    scalar, batched, preparation = _inputs(dimension, capacity)
    center, score, scale, radius, seed, search, search_scores = preparation
    # Each arm prepares/first-fits in its own numerical namespace, as the real
    # endpoint does. Cross-arm hashes require identical resulting inputs. Using
    # frozen setup in the candidate would unfairly duplicate its helper caches.
    cfg = setup.SequentialMapCovarianceConfig(refinement_geometry_policy='factor_correlation',
        structured_max_factors=1 if dimension == 3 else 2,
        structured_holdout_score_relative_rmse=.001)
    data, _ = setup._structured_factor_fit_data(scalar, center, score, scale,
        dimension=dimension, fresh_sample_count=4 * dimension, radius=float(radius),
        seed=tuple(seed.numpy().tolist()), search_theta=search, search_scores=search_scores,
        reuse_search_scores=True, evaluations=0, batched_value_and_score_fn=batched)
    first_fit = setup._fit_factor_from_data(data, factor_count=1, config=cfg)
    assert (first_fit['status'] == 'usable') == (dimension == 3)
    numerical = data['_native_factor_data']
    stages['prepared'] = _memory()
    source = (checkpoint.sources['bayesfilter/inference/sequential_map_covariance.py']
        if arm == 'before' else Path(current.__file__).read_text())
    run, excerpt = _excerpt(frozen if arm == 'before' else current, source, old=arm == 'before')
    programs = []
    if arm != 'before':
        def attempts(*args):
            program = native.attempts_program(*args, jit_compile=arm == 'xla')
            programs.append(program)
            return program

        run.__globals__['attempts_program'] = attempts
        run.__globals__['second_factor_program'] = lambda *args: provider.second_factor_program(
            *args, jit_compile=arm == 'xla')
    if arm == 'graph':
        def materialize(computed, config, dimension, active, holdout):
            assert int(computed['input_status']) == 0
            return factor._factor_result_from_computed(computed['fit'], config,
                dimension, active, holdout, False)

        run.__globals__['_factor_result_from_native'] = materialize
    args = (scalar, dimension, cfg, center, float(scalar(center)[0]), score,
        scale, float(radius), 2, first_fit, data)
    input_hashes = {key: hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest()
        for key, value in numerical.items() if tf.is_tensor(value)}
    input_hashes['first_fit'] = hashlib.sha256(json.dumps(first_fit, sort_keys=True).encode()).hexdigest()
    stages['built'] = _memory()
    samples, first = [], None
    for index in range(warm_calls + 1):
        tf.config.experimental.reset_memory_stats('GPU:0')
        started = time.perf_counter()
        result = current._json_ready(run(*args))
        elapsed = time.perf_counter() - started
        if index == 0:
            first = result
        else:
            assert result == first
        samples.append({'seconds': elapsed, 'memory': _memory()})
    stages['measured'] = _memory()
    assert first['history'][0]['proposal_attempts'][-1]['proposal_evaluated']
    assert len(first['history'][0]['proposal_attempts']) == (1 if dimension == 3 else 2)
    traces, nodes, hlo, operands = None, None, '', None
    if arm != 'before':
        assert len({id(program) for program in programs}) == 1
        program = programs[0]
        traces = program.experimental_get_tracing_count()
        assert traces == 1
        graph = program.get_concrete_function().graph.as_graph_def()
        nodes = len(graph.node) + sum(len(function.node_def) for function in graph.library.function)
        if arm == 'graph':
            assert not any(function.attr['_XlaMustCompile'].b for function in graph.library.function
                if '_XlaMustCompile' in function.attr)
        else:
            second_inputs = (() if dimension == 3 else tuple(numerical[key] for key in (
                'center_score_z', 'training_offsets_z', 'training_scores_z', 'holdout_offsets_z',
                'holdout_scores_z', 'training_weights', 'active_training_rows')))
            first_precision = (tf.constant(first_fit['projected_precision_z'], D) if dimension == 3
                else tf.zeros([dimension, dimension], D))
            native_args = (center, scalar(center)[0], score, scale, radius, tf.constant(2),
                tf.constant(dimension == 3), first_precision, second_inputs)
            hlo = program.experimental_get_compiler_ir(*native_args)(stage='hlo')
            entry = hlo[hlo.rfind('\nENTRY '):]
            operands = len(re.findall(r'\bparameter\((\d+)\)', entry))
            assert operands == (9 if dimension == 3 else 17)  # target precision, plus second-fit guard at D5
    baseline_graph = None
    if arm == 'before' and dimension == 5:
        factor_cfg = frozen.FactorCorrelationGeometryConfig(factor_count=2,
            max_condition_number=cfg.max_condition_number,
            holdout_score_relative_rmse=cfg.structured_holdout_score_relative_rmse)
        program = frozen.structured_fit_data_program(5, 10 + capacity, 10, factor_cfg)
        second_inputs = tuple(numerical[key] for key in ('center_score_z', 'training_offsets_z',
            'training_scores_z', 'holdout_offsets_z', 'holdout_scores_z', 'training_weights', 'active_training_rows'))
        graph = program.get_concrete_function().graph.as_graph_def()
        baseline_graph = len(graph.node) + sum(len(function.node_def) for function in graph.library.function)
        hlo = program.experimental_get_compiler_ir(*second_inputs)(stage='hlo')
    source_hashes = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in checkpoint.hashes()}
    for module in (native, provider):
        path = str(Path(module.__file__).relative_to(Path.cwd()))
        source_hashes[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    source_hashes[str(Path(inspect.getfile(_inputs)).relative_to(Path.cwd()))] = hashlib.sha256(
        Path(inspect.getfile(_inputs)).read_bytes()).hexdigest()
    report = {'role': 'descriptive_complete_attempt_block_cost', 'checkpoint': checkpoint.revision,
        'arm': arm, 'dimension': dimension, 'capacity': capacity,
        'jit_compile': None if arm == 'before' else arm == 'xla',
        'non_jit_role': 'explicit_graph_reference_exception' if arm == 'graph' else None,
        'execution_scope': 'original_host_controller_with_XLA_fitter_and_proposal' if arm == 'before'
            else 'native_attempt_loop_including_real_second_fit_and_complete_public_records',
        'optimizer_max_iterations': 200, 'input_sha256': input_hashes,
        'setup_namespace': 'frozen' if arm == 'before' else 'current',
        'baseline_source_sha256': checkpoint.hashes(), 'source_sha256': source_hashes,
        'excerpt_sha256': hashlib.sha256(excerpt.encode()).hexdigest(),
        'stages': stages, 'samples': samples, 'result': first,
        'trace_count': traces, 'graph_nodes': nodes, 'hlo_bytes': len(hlo.encode()), 'runtime_operands': operands,
        'baseline_second_fit_graph_nodes': baseline_graph, 'warm_calls': warm_calls,
        'timing_scope': 'exact_attempt_block_including_all_output_materialization_after_identical_first_fit',
        'nonclaims': ['One fresh process per arm; no statistical ranking or terminal repeats.',
            'Preparation and first fitting are shared setup, not measured attempt operations.',
            'Graph/XLA inherited fitter discrepancies retain unchanged 1e-10 comparison gates.',
            'Outer refinement and terminal-fit lifecycle remain separate pending repairs.']}
    directory = Path(request.config.getoption('xmlpath')).parent
    with (directory / 'attempts-memory.json').open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    with (directory / 'attempts-excerpt.py').open('x') as handle:
        handle.write(excerpt)
    if hlo:
        with (directory / 'attempts.hlo').open('x') as handle:
            handle.write(hlo)
