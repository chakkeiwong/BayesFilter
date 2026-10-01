"""Explanatory saved-HLO and RNG component diagnostics; no runtime promotion."""

import collections
import hashlib
import json
import re
import statistics
import time
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.ops.ledh_random_compat_tf import (
    philox_box_muller_normal,
    philox_replication_state,
    seeded_value_inputs,
)
from scripts.analyze_filter_repair_locator_optimized_hlo import comparison, computations
from tests.test_filter_repair_ledh_seeded_cost import _rss

RAW = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')
DTYPE = tf.float64


def _save(request, name, payload):
    directory = Path(request.config.getoption('xmlpath')).parent
    (directory / (name + '.json')).write_text(json.dumps(payload, indent=2) + '\n')
    return directory


def _hlo_census(number):
    directory = RAW / f'run-{number:05d}'
    result = json.loads((directory / 'streaming-paired-cost.json').read_text())
    path = directory / 'paired-optimized_hlo.txt'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == result['graph']['hlo']['optimized_hlo']['sha256']
    lines = path.read_text().splitlines()
    records = computations(lines)
    by_name = {record['name']: record for record in records}
    counts = collections.Counter()
    for record in records:
        counts.update(record['opcodes'])
    entry = next(record for record in records if record['entry'])
    loops = []
    for number_in_file in range(entry['line'], entry['end_line']):
        line = lines[number_in_file-1]
        if ' while(' not in line:
            continue
        body = re.search(r'body=%([^, ]+)', line).group(1)
        descendants = set()

        def visit(name, seen=descendants):
            if name not in seen:
                seen.add(name)
                for child in by_name[name]['callees']:
                    visit(child, seen)

        visit(body)
        nested = collections.Counter()
        random_anchors = []
        for name in sorted(descendants):
            record = by_name[name]
            nested.update(record['opcodes'])
            for location in range(record['line'], record['end_line']):
                instruction = lines[location-1]
                if 'op_type="StatelessRandomUniformFullIntV2"' in instruction:
                    random_anchors.append({'function': name, 'line': location})
        loops.append({'line': number_in_file, 'body': body,
            'observation_loop_shape': f"f64[{result['shape']['T']}]" in line and 'f64[64,2,2]' in line,
            'rng_only_loop_shape': f"f64[{result['shape']['T']},64,2]" in line and 'f64[64,2,2]' not in line,
            'unique_reachable_computations': len(descendants),
            'unique_reachable_opcode_counts': dict(nested),
            'philox_metadata_anchors': random_anchors})
    report = {'run': number, 'arm': result['arm'], 'horizon': result['shape']['T'],
        'hlo_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'computation_count': len(records), 'unique_opcode_counts': dict(counts),
        'entry_loops': loops,
        'limitation': 'Static unique computation counts, not executed multiplicities or timing attribution.'}
    return records, report


def test_saved_hlo_placement_profile(request):
    reports = []
    for buffered, streaming in ((4729, 4730), (4731, 4732)):
        left, left_report = _hlo_census(buffered)
        right, right_report = _hlo_census(streaming)
        reports.append({'buffered': left_report, 'streaming': right_report,
            'structural_comparison': comparison(left, right)})
        for report in (left_report, right_report):
            assert sum(loop['observation_loop_shape'] for loop in report['entry_loops']) == 1
    _save(request, 'streaming-hlo-profile', {'comparisons': reports,
        'nonclaims': ['Opcode/metadata placement alone does not establish the slowdown cause.']})


def _component(horizon, arm, *, clouds=False):
    @tf.function(input_signature=[tf.TensorSpec([1], tf.uint32)], jit_compile=True, autograph=False)
    def owner(entropy):
        state = philox_replication_state(entropy)
        if arm == 'buffered':
            initial, processes, _ = seeded_value_inputs(entropy, tf.constant([17], tf.uint32),
                horizon=horizon, particle_count=64, dimension=2, dtype=DTYPE)
            # Final state is independently determined by one initial and T process draws.
            low = state[0] + tf.constant((horizon+1)*256*128, tf.uint64)
            final = tf.stack([low, state[1]+tf.cast(low<state[0], tf.uint64), state[2]])
            summaries = tf.reduce_sum(processes, axis=1)
        else:
            initial, state = philox_box_muller_normal(state, (64, 2), DTYPE)
            output = tf.zeros([horizon, 64, 2] if clouds else [horizon, 2], DTYPE)

            def body(index, state, output):
                draw, state = philox_box_muller_normal(state, (64, 2), DTYPE)
                value = draw if clouds else tf.reduce_sum(draw, axis=0)
                return index+1, state, tf.tensor_scatter_nd_update(output, [[index]], [value])

            _, final, output = tf.while_loop(lambda index, *_: index < horizon, body,
                (tf.constant(0), state, output), parallel_iterations=1, maximum_iterations=horizon)
            if clouds:
                processes = output
            else:
                summaries = output
        return initial, processes if clouds else summaries, final
    return owner


def _sync(values):
    return tuple(value.numpy() for value in values)


@pytest.mark.parametrize('horizon,arm', [(h, a) for h in (32, 128) for a in ('buffered', 'streaming')])
def test_rng_component_profile(horizon, arm, request):
    owner = _component(horizon, arm)
    argument = tf.constant([123], tf.uint32)
    before = _rss()
    start = time.perf_counter()
    result = _sync(owner(argument))
    cold = time.perf_counter() - start
    after_cold = _rss()
    for _ in range(3):
        _sync(owner(argument))
    timings = []
    for _ in range(30):
        start = time.perf_counter()
        repeated = _sync(owner(argument))
        timings.append(time.perf_counter()-start)
    after_warm = _rss()
    for actual, expected in zip(repeated, result, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert owner.experimental_get_tracing_count() == 1
    graph = owner.get_concrete_function().graph.as_graph_def()
    nodes = [*graph.node, *(node for fn in graph.library.function for node in fn.node_def)]
    assert not {node.op for node in nodes} & {'PyFunc', 'EagerPyFunc', 'XlaHostCompute'}
    report = {'arm': arm, 'horizon': horizon, 'cold_seconds': cold, 'warm_seconds': timings,
        'warm_median_seconds': statistics.median(timings),
        'rss': {'before': before, 'after_cold': after_cold, 'after_warm': after_warm},
        'scope': 'RNG component only; buffered materialization versus streaming summaries',
        'device': owner(argument)[0].device, 'trace_count': 1}
    directory = _save(request, 'streaming-rng-profile', report)
    hlo = owner.experimental_get_compiler_ir(argument)(stage='optimized_hlo')
    (directory / 'streaming-rng-optimized_hlo.txt').write_text(hlo)
    report['hlo_sha256'] = hashlib.sha256(hlo.encode()).hexdigest()
    report['process_buffer_shape_occurrences'] = hlo.count(f'f64[{horizon},64,2]')
    buffered, streaming = _component(horizon, 'buffered', clouds=True), _component(horizon, 'streaming', clouds=True)
    for seed in (123, 124):
        operand = tf.constant([seed], tf.uint32)
        first, second = _sync(buffered(operand)), _sync(streaming(operand))
        for lhs, rhs in zip(first, second, strict=True):
            np.testing.assert_array_equal(lhs, rhs)
        summary = _sync(owner(operand))
        np.testing.assert_allclose(summary[1], np.sum(first[1], axis=1), atol=1e-12, rtol=1e-12)
        np.testing.assert_array_equal(summary[2], first[2])
    assert owner.experimental_get_tracing_count() == 1
    report['complete_cloud_and_state_exact_for_seeds'] = [123, 124]
    _save(request, 'streaming-rng-profile', report)
