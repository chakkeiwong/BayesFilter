"""Descriptive fresh-process cost of the complete native proposal subprogram."""

import hashlib
import inspect
import json
import subprocess
import time
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference import sequential_map_covariance as sequential
from bayesfilter.inference import sequential_preparation_tf as preparation
from bayesfilter.inference import sequential_proposal_tf as proposal
from bayesfilter.inference import sequential_structured_preparation_tf as structured
from tests.test_filter_repair_factor_capacity import _memory
from tests.test_filter_repair_sequential_proposal import _arguments, _expected, _target
from tests.test_filter_repair_sequential_proposal import frozen as _frozen


@pytest.mark.parametrize('dimension', [3, 5])
@pytest.mark.parametrize('arm', ['before', 'graph', 'xla'])
def test_proposal_memory(dimension, arm, request):
    frozen = _frozen.__wrapped__() if arm == 'before' else None
    stages = {'before': _memory()}
    arguments = _arguments(dimension)
    program = None if arm == 'before' else proposal.proposal_program(
        _target, dimension, 'resolvable_decrease', True, jit_compile=arm != 'graph')
    hashes = [hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest() for value in arguments]
    stages['built'] = _memory()
    samples, first = [], None
    for index in range(21):
        tf.config.experimental.reset_memory_stats('GPU:0')
        started = time.perf_counter()
        output = (_expected(frozen, _target, arguments, 'resolvable_decrease', True)
            if arm == 'before' else program(*arguments))
        result = {key: value.numpy().tolist() if tf.is_tensor(value) else value for key, value in output.items()}
        elapsed = time.perf_counter() - started
        if first is None:
            first = result
        else:
            assert result == first
        samples.append({'seconds': elapsed, 'memory': _memory()})
    stages['measured'] = _memory()
    hlo, nodes, traces = '', None, None
    if program is not None:
        traces = program.experimental_get_tracing_count()
        assert traces == 1
        graph = program.get_concrete_function().graph.as_graph_def()
        nodes = len(graph.node) + sum(len(function.node_def) for function in graph.library.function)
        if arm == 'graph':
            assert not any(function.attr.get('_XlaMustCompile', None).b
                for function in graph.library.function if '_XlaMustCompile' in function.attr)
        else:
            hlo = program.experimental_get_compiler_ir(*arguments)(stage='hlo')
    sources = {str(Path(item.__file__).relative_to(Path.cwd())):
        hashlib.sha256(Path(item.__file__).read_bytes()).hexdigest()
        for item in (sequential, preparation, proposal, structured, inspect.getmodule(_target))}
    report = {'role': 'descriptive_proposal_checkpoint_cost', 'checkpoint': 'b3334646',
        'arm': arm, 'dimension': dimension, 'jit_compile': None if arm == 'before' else arm == 'xla',
        'execution_scope': 'mixed_host_with_XLA_trust_solver' if arm == 'before' else 'complete_numerical_proposal',
        'baseline_source_sha256': {name: hashlib.sha256(subprocess.check_output(['git', 'show', 'b3334646:' + name])).hexdigest()
            for name in ('bayesfilter/inference/sequential_map_covariance.py', 'bayesfilter/inference/sequential_preparation_tf.py')},
        'non_jit_role': 'explicit_graph_reference_exception' if arm == 'graph' else None,
        'input_sha256': hashes, 'source_sha256': sources, 'stages': stages, 'samples': samples,
        'result': first, 'hlo_bytes': len(hlo.encode()), 'graph_nodes': nodes, 'trace_count': traces,
        'timing_scope': 'complete_proposal_and_every_numeric_field_materialized',
        'nonclaims': ['Outer refinement/escalation are separate pending repairs.',
            'One process per arm; no statistical ranking or terminal repeat evidence.']}
    directory = Path(request.config.getoption('xmlpath')).parent
    with (directory / 'proposal-memory.json').open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
