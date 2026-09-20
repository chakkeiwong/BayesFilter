"""Diagnostic stage attribution of complete lifecycle host-memory growth."""

import json
import re
import resource
import time
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference import sequential_lifecycle_tf as native
from bayesfilter.inference import sequential_map_covariance as current
from bayesfilter.inference.sequential_refinement_tf import refinement_program
from bayesfilter.inference.sequential_terminal_tf import terminal_program
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_factor_capacity import _memory
from tests.test_filter_repair_lifecycle_actual import _excerpts, _materialize
from tests.test_filter_repair_structured_memory import _inputs

D = tf.float64


@pytest.mark.parametrize('dimension,search_count', [(3, 4), (5, 32)])
def test_lifecycle_compilation_memory_stages(dimension, search_count, request):
    # Keep baseline namespace construction identical to the cost measurement.
    checkpoint = FrozenCheckpoint('cfbc32d2', 'lifecycle_profile')
    _, finish = _excerpts(checkpoint)
    stages = []
    start = time.perf_counter()

    def observe(stage):
        rollup = {row.split()[0].rstrip(':'): int(row.split()[1]) * 1024
            for row in Path('/proc/self/smaps_rollup').read_text().splitlines()
            if row.startswith(('Rss:', 'Pss:', 'Private_Clean:', 'Private_Dirty:', 'Swap:'))}
        stages.append({'stage': stage, 'elapsed_seconds': time.perf_counter() - start,
            'memory': _memory(), 'host_map_count': len(Path('/proc/self/maps').read_text().splitlines()),
            'ru_maxrss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            'smaps_rollup_bytes': rollup})

    observe('before_preparation')
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
    observe('prepared')
    refine = refinement_program(scalar, batched, dimension, cfg, search_count)
    observe('refinement_factory')
    terminal = terminal_program(scalar, batched, dimension, cfg)
    observe('terminal_factory')
    program = native.lifecycle_program(refine, terminal, dimension, cfg, search_count)
    observe('lifecycle_factory')
    concrete = program.get_concrete_function()
    observe('outer_graph_traced')
    tf.config.experimental.reset_memory_stats('GPU:0')
    raw = program(*arguments)
    int(raw['status'])
    observe('cold_numerical_complete_before_reporting')
    progress = []
    result = _materialize(raw, center, value, score, scale, cfg, finish, progress).payload()
    observe('complete_report_materialized')
    first = {'result': result, 'progress': progress}
    del raw, result, progress
    observe('raw_outputs_released')
    for index in range(20):
        raw = program(*arguments)
        int(raw['status'])
        progress = []
        result = _materialize(raw, center, value, score, scale, cfg, finish, progress).payload()
        assert {'result': result, 'progress': progress} == first
        del raw, result, progress
        observe(f'warm_complete_{index + 1}')
    # Serialization and compiler IR inspection happen only after memory samples.
    graphs = {}
    for name, function in [('refinement', refine), ('terminal', terminal), ('lifecycle', program)]:
        graph = function.get_concrete_function().graph.as_graph_def()
        graphs[name] = {'nodes': len(graph.node), 'functions': len(graph.library.function),
            'function_nodes': sum(len(item.node_def) for item in graph.library.function),
            'serialized_bytes': graph.ByteSize(), 'trace_count': function.experimental_get_tracing_count()}
    hlo = program.experimental_get_compiler_ir(*arguments)(stage='hlo')
    optimized = program.experimental_get_compiler_ir(*arguments)(stage='optimized_hlo')
    assert program.experimental_get_tracing_count() == 1
    directory = Path(request.config.getoption('xmlpath')).parent
    for filename, content in [('lifecycle-profile.hlo', hlo), ('lifecycle-profile-optimized.hlo', optimized)]:
        with (directory / filename).open('x') as handle:
            handle.write(content)
    outputs = tf.nest.flatten(concrete.structured_outputs)
    report = {'role': 'explanatory_host_compilation_stage_attribution', 'dimension': dimension,
        'search_count': search_count, 'optimizer_max_iterations': 200, 'attempt_limit': cfg.max_attempts,
        'checkpoint': checkpoint.revision, 'frozen_source_sha256': checkpoint.hashes(),
        'stages': stages, 'graphs': graphs, 'result': first,
        'output_count': len(outputs),
        'logical_output_bytes': sum(item.shape.num_elements() * item.dtype.size for item in outputs),
        'hlo_bytes': len(hlo.encode()), 'optimized_hlo_bytes': len(optimized.encode()),
        'optimized_operations': {name: len(re.findall(r'= [^\n]*?\b' + re.escape(name) + r'\(', optimized))
            for name in ('while', 'conditional', 'dot', 'custom-call', 'fusion')},
        'nonclaims': ['Explicit tracing shifts timing boundaries; this explains memory, not comparative speed.',
            'Warm samples do not establish native executable-cache eviction or arbitrary-capacity memory bounds.',
            'Complete numerical comparisons and public/external integration remain separate gates.']}
    with (directory / 'lifecycle-profile.json').open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
