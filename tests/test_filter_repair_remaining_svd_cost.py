"""Diagnostic matched costs of the five-consumer SVD repair composition."""

import csv
import gc
import os
import subprocess
import sys
import time
import types
import weakref
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.program_cache_scope import independent_trace_scope
from scripts.filter_repair_cost_provenance import GPUProcessMonitor
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_block_capture import stable_hlo
from tests.test_filter_repair_gap_diagnostics import memory_snapshot
from tests.test_filter_repair_geometry_control import clean, save

D = tf.float64
BASELINE = '9696666e4'
MODULES = ('block_score_geometry_tf', 'quadratic_geometry', 'score_curvature_tf',
           'sequential_score_fit_tf')


class CostCheckpoint(FrozenCheckpoint):
    def load(self, name):
        if name != 'bayesfilter.runtime' or name in self.modules:
            return super().load(name)
        # This checkpoint's fixed-center module imports an HMC artifact helper.
        # Execute the exact lazy runtime export table, binding its import_module
        # to this frozen loader so no package export resolves live project code.
        path = 'bayesfilter/runtime/__init__.py'
        source = subprocess.check_output(['git', 'show', f'{self.revision}:{path}'],
            cwd=Path(__file__).resolve().parents[1], text=True)
        module = types.ModuleType(self.prefix + '.' + name)
        module.__path__ = []
        module.__file__ = f'{self.revision}:{path}'
        self.modules[name] = module
        self.sources[path] = source
        sys.modules[module.__name__] = module
        exec(compile(source, module.__file__, 'exec'), module.__dict__)  # noqa: S102 - exact frozen diagnostic code
        module.import_module = self.load
        return module


def source_functions(prior):
    checkpoint = CostCheckpoint(BASELINE, 'remaining_svd_cost')
    original = [checkpoint.load('bayesfilter.inference.' + name) for name in MODULES]
    condition = checkpoint.load('bayesfilter.ops.qr_lstsq_tf').condition_number
    if prior:
        modules = original
    else:
        from bayesfilter.inference import (
            block_score_geometry_tf,
            quadratic_geometry,
            score_curvature_tf,
            sequential_score_fit_tf,
        )
        from bayesfilter.ops.qr_lstsq_tf import condition_number
        modules = [block_score_geometry_tf, quadratic_geometry, score_curvature_tf,
            sequential_score_fit_tf]
        condition = condition_number
    return (modules[0]._block_fit, modules[1]._quadratic_fit_kernel,
        modules[2].fit_dense_score_precision_tf, modules[3].fit_numerics,
        condition), checkpoint


def fixture(dimension):
    rows = 2 * dimension + 1
    if dimension == 3:
        raw = np.array([[1., .2, -.3], [.4, 1., .1], [-.1, .3, 1.],
            [.7, -.5, .2], [-.3, .2, .9], [.2, .7, -.6], [-.8, .3, .2]])
        right_raw = np.array([[1., .4, -.2], [.3, 1., .5], [-.1, .2, 1.]])
        separated = np.array([3., 1.5, .7])
        precision = np.diag([1.3, 1.7, .8])
    else:
        indices = np.arange(rows * dimension).reshape(rows, dimension)
        raw = np.sin((indices + 1.) ** 2 / 11.) + np.cos(.7 * indices + .3)
        right_raw = np.cos(np.arange(dimension ** 2).reshape(dimension, dimension) + .5)
        right_raw += np.eye(dimension)
        separated = np.array([3., 2.3, 1.5, 1.1, .7])
        precision = np.diag([1.3, 1.7, 1.1, 1.5, .8])
    left, right = np.linalg.qr(raw)[0], np.linalg.qr(right_raw)[0]
    near_tied = 1. + np.arange(dimension // 2, -dimension // 2, -1) * 1e-8
    assert near_tied.shape == (dimension,)
    return left, right, separated, near_tied, precision


def build_numerical(functions, dimension, jit):
    block, quadratic, dense, sequential, condition = functions
    rows = 2 * dimension + 1
    @tf.function(input_signature=[tf.TensorSpec([rows, dimension], D),
        tf.TensorSpec([dimension, dimension], D)],
        jit_compile=jit, autograph=False)
    def evaluate(offsets, precision):
        response = tf.matmul(offsets, precision)
        zero = tf.zeros([dimension], D)
        block_precision, block_report, block_status = block(offsets, response,
            tf.constant(0., D), tf.constant(1e8, D))
        return {'condition': condition(offsets, jit_compile=jit),
            'dense': dense(zero, offsets, -response),
            'block_precision': block_precision, 'block_report': block_report,
            'block_status': block_status,
            'sequential': sequential(offsets, -response, zero, tf.ones([dimension], D),
                tf.constant(0., D), tf.constant(1e-8, D), tf.constant(1e8, D),
                tf.constant(.05, D), training_indices=tuple(range(rows - 2)),
                holdout_indices=(rows - 2, rows - 1), jit_compile=jit),
            'quadratic': quadratic(offsets, -.5 * tf.reduce_sum(offsets * response, axis=1),
                -response, tf.eye(dimension, dtype=D)[:, :dimension-1], zero,
                tf.constant(1e-8, D), tf.constant(1e8, D), use_xla_svd=jit)}
    # This diagnostic shape-only composition contains the inherited COD custom
    # gradients. Their registry may retain this bounded primitive graph. Trace
    # before creating any caller, callback or resource so it cannot retain them.
    with independent_trace_scope():
        evaluate.get_concrete_function()
    return evaluate


class PrecisionSource:
    def __init__(self, precision):
        self.resource = tf.Variable(precision, dtype=D, trainable=False)

    def read(self):
        return self.resource.read_value()


def build_endpoint(numerical, callback, dimension, jit):
    @tf.function(input_signature=[tf.TensorSpec([2 * dimension + 1, dimension], D)],
        jit_compile=jit, autograph=False)
    def evaluate(offsets):
        return numerical(offsets, callback())
    return evaluate


def checks(result, offsets, precision):
    dimension = precision.shape[0]
    rank = dimension * (dimension + 1) // 2
    expected_condition = np.linalg.cond(offsets)
    def near(actual, expected):
        return bool(np.allclose(actual, expected, rtol=1e-10, atol=1e-10))
    return {'cod_condition': near(result['condition'], expected_condition),
        'dense_condition': near(result['dense']['design_condition'], expected_condition),
        'dense_rank': bool(result['dense']['design_rank'] == dimension),
        'dense_precision': near(result['dense']['raw_precision'], precision),
        'block_ranks': bool(np.array_equal(result['block_report'][:2], [rank, dimension])),
        'block_status': bool(result['block_status'] == 0),
        'block_precision': near(result['block_precision'], precision),
        'sequential_rank': bool(result['sequential']['rank'] == rank),
        'sequential_status': bool(result['sequential']['status'] == 1),
        'sequential_precision': near(result['sequential']['projected_precision_z'], precision),
        'quadratic_rank': bool(result['quadratic']['score_design_rank'] == dimension),
        'quadratic_resolved': bool(result['quadratic']['design_resolved']),
        'quadratic_precision': near(result['quadratic']['precision'], precision)}


def memory(gpu):
    result = memory_snapshot(gpu)
    result['mapping_count'] = len(Path('/proc/self/maps').read_text().splitlines())
    result['process_gpu_reservation_bytes'] = None
    if gpu:
        output = subprocess.check_output(['nvidia-smi',
            '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'], text=True)
        entries = [int(used) * 2 ** 20 for uuid, pid, used in
            csv.reader(output.splitlines(), skipinitialspace=True)
            if uuid == os.environ['CUDA_VISIBLE_DEVICES'] and int(pid) == os.getpid()]
        if len(entries) != 1:
            raise ValueError('Missing or ambiguous process GPU reservation')
        result['process_gpu_reservation_bytes'] = entries[0]
    return result


@pytest.mark.parametrize('dimension', [3, 5])
@pytest.mark.parametrize('arm', ['prior_graph', 'prior_xla', 'after_graph', 'after_xla'])
def test_remaining_consumer_svd_cost(arm, dimension, request):
    functions, checkpoint = source_functions(arm.startswith('prior'))
    left, right, separated, near_tied, precision = fixture(dimension)
    initial_offsets = (left * separated) @ right.T
    operands = tf.constant(initial_offsets, D)
    changed = tf.constant(initial_offsets * 1.1, D)
    jit = arm.endswith('xla')
    gpu = bool(tf.config.list_logical_devices('GPU'))
    owner = PrecisionSource(precision)
    callback = owner.read
    snapshots = {'prepared': memory(gpu)}
    replay_failures = []
    with GPUProcessMonitor(gpu) as sharing:
        tick = time.perf_counter()
        numerical = build_numerical(functions, dimension, jit)
        program = build_endpoint(numerical, callback, dimension, jit)
        program.get_concrete_function()
        construction_trace_seconds = time.perf_counter() - tick
        snapshots['traced'] = memory(gpu)

        def execute(program, values):
            if gpu:
                tf.config.experimental.reset_memory_stats('GPU:0')
            tick = time.perf_counter()
            actual = tf.nest.map_structure(lambda value: np.asarray(value.numpy()), program(values))
            seconds = time.perf_counter() - tick
            return actual, {'seconds': seconds, 'memory': memory(gpu)}

        initial, cold = execute(program, operands)
        samples = []
        for index in range(20):
            replay, cost = execute(program, operands)
            samples.append(cost)
            if not all(np.array_equal(a, b, equal_nan=True) for a, b in
                zip(tf.nest.flatten(initial), tf.nest.flatten(replay), strict=True)):
                replay_failures.append(index)
        owner.resource.assign(precision * 1.05)
        second, changed_cost = execute(program, changed)
        owner.resource.assign(precision)
        returned, returned_cost = execute(program, operands)
        if not all(np.array_equal(a, b, equal_nan=True) for a, b in
            zip(tf.nest.flatten(initial), tf.nest.flatten(returned), strict=True)):
            replay_failures.append('returned')
        snapshots['completed'] = memory(gpu)

    # The multi-scale numerical screen reuses the measured program after costs.
    # A failed baseline is retained as failure evidence and never speed-ranked.
    scale_records = []
    for name, spectrum in (('separated', separated), ('near_tied', near_tied)):
        for magnitude in (1., 1e-4, 1e-10, 1e4):
            offsets = (left * spectrum) @ right.T * magnitude
            actual = tf.nest.map_structure(lambda value: np.asarray(value.numpy()), program(tf.constant(offsets, D)))
            scale_records.append({'spectrum': name, 'magnitude': magnitude, 'offsets': offsets,
                'actual': actual, 'checks': checks(actual, offsets, precision)})
    initial_checks = checks(initial, initial_offsets, precision)
    changed_checks = checks(second, initial_offsets * 1.1, precision * 1.05)
    numerical_passed = all(initial_checks.values()) and all(changed_checks.values()) and all(
        all(row['checks'].values()) for row in scale_records)
    graph = program.get_concrete_function().graph
    definition = graph.as_graph_def()
    operations = {node.op for nodes in (definition.node, *(fn.node_def for fn in definition.library.function)) for node in nodes}
    program_info = {'trace_count': program.experimental_get_tracing_count(),
        'nodes': len(definition.node) + sum(len(fn.node_def) for fn in definition.library.function),
        'bytes': definition.ByteSize(), 'host_callbacks': sorted(operations & {'PyFunc', 'PyFuncStateless', 'EagerPyFunc'})}
    if jit:
        hlo = program.experimental_get_compiler_ir(operands)(stage='hlo')
        other = program.experimental_get_compiler_ir(changed)(stage='hlo')
        program_info.update(hlo_bytes=len(hlo.encode()), hlo_unchanged=stable_hlo(hlo) == stable_hlo(other))
    references = {'program': weakref.ref(program), 'graph': weakref.ref(graph),
        'owner': weakref.ref(owner), 'callback': weakref.ref(callback),
        'resource': weakref.ref(owner.resource)}
    primitive = {'trace_count': numerical.experimental_get_tracing_count(),
        'capture_count': len(numerical.get_concrete_function().captured_inputs),
        'role': 'Reusable shape-only diagnostic composition; inherited COD registry retention is bounded by worker lifetime.'}
    primitive_reference = weakref.ref(numerical.get_concrete_function().graph)
    del graph, program, execute, owner, callback, numerical
    gc.collect()
    released = {name: reference() is None for name, reference in references.items()}
    primitive['graph_released'] = primitive_reference() is None
    snapshots['released'] = memory(gpu)
    observation = sharing.payload()
    unshared = not observation['errors'] and (not gpu or len(observation['samples']) >= 2 and all(
        row['uuid'] == observation['uuid'] and row['pid'] == observation['pid']
        for sample in observation['samples'] for row in sample['processes']))
    report = {'schema': 'filter_remaining_svd_cost.v1', 'arm': arm, 'dimension': dimension,
        'baseline': BASELINE, 'original_source_sha256': checkpoint.hashes(), 'gpu': gpu,
        'jit_compile': jit, 'construction_trace_seconds': construction_trace_seconds,
        'cold': cold, 'samples': samples, 'changed': changed_cost, 'returned': returned_cost,
        'snapshots': snapshots, 'program': program_info, 'released': released,
        'primitive': primitive, 'initial_offsets': initial_offsets,
        'changed_offsets': initial_offsets * 1.1, 'changed_precision': precision * 1.05,
        'initial': initial, 'changed_values': second, 'returned_values': returned,
        'expected_precision': precision,
        'initial_checks': initial_checks, 'changed_checks': changed_checks,
        'scale_records': scale_records, 'numerical_passed': numerical_passed,
        'replay_failures': replay_failures, 'in_run_unshared': unshared,
        'gpu_process_observation': observation,
        'timing_scope': 'Reusable actual dense/block/sequential/quadratic/COD shape-only composition plus owning resource-read endpoint and output materialization.',
        'nonclaims': ['Inaccurate baseline costs cannot support ratios.',
            'CPU and graph are explicit references; fixture costs are descriptive.',
            'Native-memory and public/actual-consumer capacity remain separate.']}
    save(request, 'remaining-svd-cost.json', clean(report))
    assert not replay_failures and unshared
    assert program_info['trace_count'] == 1 and not program_info['host_callbacks']
    assert not jit or program_info['hlo_unchanged']
    if arm != 'prior_xla':
        assert numerical_passed, [row['checks'] for row in scale_records]
    assert all(released.values()), released
    assert primitive['trace_count'] == 1 and primitive['capture_count'] == 0
