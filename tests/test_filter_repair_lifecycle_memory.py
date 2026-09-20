"""Fresh-process diagnostic costs of the complete original/native lifecycle."""

import hashlib
import json
import re
import time
from pathlib import Path
from types import FunctionType, SimpleNamespace

import pytest
import tensorflow as tf

from bayesfilter.inference import factor_correlation_geometry as factor
from bayesfilter.inference import sequential_lifecycle_tf as native
from bayesfilter.inference import sequential_map_covariance as current
from bayesfilter.inference.sequential_refinement_tf import refinement_program
from bayesfilter.inference.sequential_terminal_tf import terminal_program
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_factor_capacity import _memory
from tests.test_filter_repair_lifecycle_actual import _excerpts, _materialize
from tests.test_filter_repair_sequential_refinement import _factor_payload, _record
from tests.test_filter_repair_structured_memory import _inputs

D = tf.float64


def _graph_materializer():
    # Rebind diagnostic report helpers only, so a graph-mode fitted record is
    # labelled correctly and its status reporting does not invoke XLA.
    def materialize(computed, cfg, dimension, active, holdout):
        assert int(computed['input_status']) == 0
        return factor._factor_result_from_computed(computed['fit'], cfg, dimension, active, holdout, False)

    namespace = SimpleNamespace(**{**vars(current), '_factor_result_from_native': materialize})
    payload = FunctionType(_factor_payload.__code__, {**_factor_payload.__globals__, 'current': namespace})
    record = FunctionType(_record.__code__, {**_record.__globals__, '_factor_payload': payload})
    return FunctionType(_materialize.__code__, {**_materialize.__globals__, '_record': record})


@pytest.mark.parametrize('dimension,search_count', [(3, 4), (5, 32)])
@pytest.mark.parametrize('arm', ['before', 'graph', 'xla'])
def test_complete_lifecycle_memory(dimension, search_count, arm, request):
    checkpoint = FrozenCheckpoint('cfbc32d2', 'lifecycle_cost')
    original, finish = _excerpts(checkpoint)
    stages = {'before_preparation': _memory()}
    started = time.perf_counter()
    scalar, batched, _ = _inputs(dimension, search_count)
    cfg = current.SequentialMapCovarianceConfig(locator_policy='center_first',
        refinement_geometry_policy='factor_correlation', structured_max_factors=1 if dimension == 3 else 2,
        reuse_search_scores=True, structured_holdout_score_relative_rmse=.001,
        max_attempts=2, search_sample_count=search_count, terminal_sample_count=24,
        max_exact_evaluations=512, initial_radius=.25, terminal_score_max_abs=1e-5,
        record_refinement_movement_diagnostics=True)
    center = tf.linspace(tf.constant(.002, D), tf.constant(.004, D), dimension)
    value, score = scalar(center)
    scale = tf.ones([dimension], D)
    arguments = (center, value, score, scale, tf.constant(1, tf.int64))
    setup_seconds = time.perf_counter() - started
    stages['prepared'] = _memory()
    started = time.perf_counter()
    if arm != 'before':
        refine = refinement_program(scalar, batched, dimension, cfg, search_count, jit_compile=arm == 'xla')
        terminal = terminal_program(scalar, batched, dimension, cfg, jit_compile=arm == 'xla')
        program = native.lifecycle_program(refine, terminal, dimension, cfg, search_count, jit_compile=arm == 'xla')
    materialize = _graph_materializer() if arm == 'graph' else _materialize
    build_seconds = time.perf_counter() - started
    stages['built'] = _memory()

    def execute(args):
        point, point_value, point_score, point_scale, evaluations = args
        tf.config.experimental.reset_memory_stats('GPU:0')
        progress = []
        started = time.perf_counter()
        if arm == 'before':
            result = original(scalar, batched, point, float(point_value), point_score,
                point_scale, cfg, int(evaluations), progress.append).payload()
            numerical_seconds, reporting_seconds = None, None
        else:
            raw = program(*args)
            # Synchronize execution before separating report construction cost.
            int(raw['status'])
            numerical_seconds = time.perf_counter() - started
            report_started = time.perf_counter()
            result = materialize(raw, point, point_value, point_score, point_scale,
                cfg, finish, progress).payload()
            reporting_seconds = time.perf_counter() - report_started
        elapsed = time.perf_counter() - started
        observation = {'result': result, 'progress': progress}
        return observation, {'seconds': elapsed, 'numerical_seconds': numerical_seconds,
            'reporting_seconds': reporting_seconds, 'memory': _memory()}

    samples, first = [], None
    for index in range(21):
        observation, sample = execute(arguments)
        if first is None:
            first = observation
        else:
            assert observation == first
        samples.append(sample)
    stages['warm_measured'] = _memory()
    changed_inputs = []
    for multiplier in (.8, .6):
        point = center * tf.constant(multiplier, D)
        point_value, point_score = scalar(point)
        changed = (point, point_value, point_score, scale * tf.constant(1.1, D), tf.constant(7, tf.int64))
        result, cold_changed = execute(changed)
        repeated, warm_changed = execute(changed)
        assert result == repeated
        changed_inputs.append({'input_sha256': [hashlib.sha256(tf.io.serialize_tensor(item).numpy()).hexdigest()
            for item in changed], 'first_call': cold_changed, 'immediate_repeat': warm_changed, 'result': result})
    stages['measured'] = _memory()
    assert any(row.get('proposal_attempts') for row in first['result']['diagnostics']['history'])
    if dimension == 5:
        assert any(len(row.get('proposal_attempts', [])) == 2 and row['proposal_attempts'][1]['proposal_evaluated']
            for row in first['result']['diagnostics']['history'])
    traces, nodes, hlo, operands = None, None, '', None
    if arm != 'before':
        concrete = program.get_concrete_function()
        graph = concrete.graph.as_graph_def()
        nodes = len(graph.node) + sum(len(function.node_def) for function in graph.library.function)
        traces = program.experimental_get_tracing_count()
        assert traces == 1
        if arm == 'graph':
            assert not any(function.attr['_XlaMustCompile'].b for function in graph.library.function
                if '_XlaMustCompile' in function.attr)
        else:
            hlo = program.experimental_get_compiler_ir(*arguments)(stage='hlo')
            entry = hlo[hlo.rfind('\nENTRY '):]
            operands = len(re.findall(r'\bparameter\((\d+)\)', entry))
            assert operands == 5 + len(concrete.captured_inputs)
    paths = (*checkpoint.hashes(), 'bayesfilter/inference/sequential_lifecycle_tf.py',
        'bayesfilter/inference/sequential_refinement_tf.py', 'bayesfilter/inference/sequential_terminal_tf.py')
    report = {'role': 'descriptive_complete_lifecycle_checkpoint_cost', 'checkpoint': checkpoint.revision,
        'arm': arm, 'dimension': dimension, 'search_count': search_count, 'attempt_limit': cfg.max_attempts,
        'optimizer_max_iterations': 200, 'jit_compile': None if arm == 'before' else arm == 'xla',
        'non_jit_role': 'explicit_graph_reference_exception' if arm == 'graph' else None,
        'baseline_source_sha256': checkpoint.hashes(),
        'source_sha256': {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths},
        'input_sha256': [hashlib.sha256(tf.io.serialize_tensor(item).numpy()).hexdigest() for item in arguments],
        'setup_seconds': setup_seconds, 'build_seconds': build_seconds,
        'cold_total_seconds': setup_seconds + build_seconds + samples[0]['seconds'],
        'stages': stages, 'samples': samples, 'result': first, 'trace_count': traces,
        'changed_inputs': changed_inputs,
        'graph_nodes': nodes, 'hlo_bytes': len(hlo.encode()), 'runtime_operands': operands,
        'timing_scope': 'entire numerical lifecycle plus complete records and buffered event reconstruction',
        'shared_final_mass_mode': 'unchanged public XLA mass helper, outside lifecycle numerical graph',
        'nonclaims': ['One process per arm; descriptive only, not terminal repeats.',
            'Current test report materialization is diagnostic, not the final public integration.',
            'Strict graph/XLA comparisons remain mandatory; inherited failures are not waived.',
            'Buffered event records do not qualify live callback or external timeout behavior.']}
    directory = Path(request.config.getoption('xmlpath')).parent
    with (directory / 'lifecycle-memory.json').open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    if hlo:
        with (directory / 'lifecycle-memory.hlo').open('x') as handle:
            handle.write(hlo)
