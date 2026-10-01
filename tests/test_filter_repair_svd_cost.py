"""Diagnostic matched graph/XLA costs of the actual repaired SRUKF route."""

import hashlib
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.nonlinear import rectangular_srukf_tf as current
from scripts.filter_repair_cost_provenance import GPUProcessMonitor
from tests.test_filter_repair_block_capture import stable_hlo
from tests.test_filter_repair_gap_diagnostics import memory_snapshot
from tests.test_filter_repair_geometry_control import clean, save
from tests.test_filter_repair_srukf_scale import independent_kalman

D = tf.float64
BASELINE = 'f4ea46dde'


def original_route():
    repo = Path(__file__).resolve().parents[1]
    sources = {}

    def module_at(path, name):
        source = subprocess.check_output(['git', 'show', f'{BASELINE}:{path}'], cwd=repo)
        spec = importlib.util.spec_from_loader(name, loader=None)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        exec(compile(source, f'{BASELINE}:{path}', 'exec'), module.__dict__)  # noqa: S102
        sources[path] = hashlib.sha256(source).hexdigest()
        return module

    factors = module_at('bayesfilter/linear/rectangular_factor_tf.py', 'svd_cost_original_factors')
    route = module_at('bayesfilter/nonlinear/rectangular_srukf_tf.py', 'svd_cost_original_route')
    route.batched_direct_stack_svd_factor = factors.batched_direct_stack_svd_factor
    route.batched_direct_support_conditional = factors.batched_direct_support_conditional
    # Value-route cubature is unchanged; pin that fact rather than allow a
    # current-code dependency to silently alter the numerical authority.
    dependency = 'bayesfilter/nonlinear/factor_srukf_tf.py'
    original = subprocess.check_output(['git', 'show', f'{BASELINE}:{dependency}'], cwd=repo)
    assert original == (repo / dependency).read_bytes()
    sources[dependency] = hashlib.sha256(original).hexdigest()
    return route, sources


@pytest.mark.parametrize('horizon', [1, 3])
@pytest.mark.parametrize('arm', ['prior_graph', 'prior_xla', 'after_graph', 'after_xla'])
def test_complete_srukf_svd_cost(arm, horizon, request):
    original, sources = original_route()
    route = original if arm.startswith('prior') else current
    jit = arm.endswith('xla')
    transition = np.array([[.8, .1, -.05], [.04, .7, .15], [-.02, .06, .9]])
    observation = np.array([[2., 1., -.5], [1., 3., .25], [-.5, .25, 1.]])
    prior, process, noise = np.diag([.5, .3, .4]), np.diag([.2, .1, .15]), np.diag([.15, .1, .2])
    points = np.array([[.1, -.2, .3], [.2, .1, -.1], [-.1, .15, .2]])[:horizon]
    transition_tf, observation_tf = tf.constant(transition, D), tf.constant(observation, D)
    model = route.TFRectangularSRUKFModel(tf.zeros([1, 3], D), tf.constant(prior[None], D),
        tf.constant(process[None], D), tf.constant(noise[None], D),
        lambda state, noise: tf.linalg.matmul(state, transition_tf, transpose_b=True) + noise,
        lambda state: tf.linalg.matmul(state, observation_tf, transpose_b=True))
    operands, changed = tf.constant(points[None], D), tf.constant((points + .02)[None], D)
    gpu = bool(tf.config.list_logical_devices('GPU'))
    snapshots = {'prepared': memory_snapshot(gpu)}
    with GPUProcessMonitor(gpu) as sharing:
        tick = time.perf_counter()

        @tf.function(input_signature=[tf.TensorSpec([1, horizon, 3], D)],
            autograph=False, jit_compile=jit)
        def evaluate(observations):
            result = route.tf_rectangular_srukf_value(observations, model, jit_compile=False)
            return {'likelihood': result.log_likelihood, 'mean': result.filtered_mean,
                'covariance': tf.matmul(result.filtered_factor, result.filtered_factor, transpose_b=True),
                'on_support': result.diagnostics['on_support'],
                'rank': result.diagnostics['minimum_observation_rank'],
                'support_residual': result.diagnostics['maximum_support_residual']}

        evaluate.get_concrete_function()
        construction_trace_seconds = time.perf_counter() - tick
        snapshots['traced'] = memory_snapshot(gpu)

        def execute(inputs):
            if gpu:
                tf.config.experimental.reset_memory_stats('GPU:0')
            tick = time.perf_counter()
            values = tf.nest.map_structure(lambda value: value.numpy(), evaluate(inputs))
            elapsed = time.perf_counter() - tick
            return values, {'seconds': elapsed, 'memory': memory_snapshot(gpu)}

        initial, cold = execute(operands)
        samples = []
        for _ in range(20):
            replay, cost = execute(operands)
            samples.append(cost)
            for name in initial:
                np.testing.assert_array_equal(replay[name], initial[name])
        second, changed_cost = execute(changed)
        returned, returned_cost = execute(operands)
        snapshots['completed'] = memory_snapshot(gpu)

    # Independent reference and IR export occur after measured observations.
    comparisons = []
    for values, data in ((initial, points), (second, points + .02)):
        likelihood, mean, covariance = independent_kalman(data, transition, observation, prior, process, noise)
        expected = {'likelihood': np.array([likelihood]), 'mean': mean[None], 'covariance': covariance[None]}
        errors = {name: float(np.max(np.abs(values[name] - expected[name]))) for name in expected}
        passed = (all(np.allclose(values[name], expected[name], rtol=1e-10, atol=1e-10) for name in expected)
            and bool(values['on_support'].all()) and bool((values['rank'] == 3).all()))
        comparisons.append({'passed': passed, 'absolute_errors': errors, 'actual': clean(values),
            'independent_reference': clean(expected)})
    graph = evaluate.get_concrete_function().graph.as_graph_def()
    graph_info = {'nodes': len(graph.node) + sum(len(f.node_def) for f in graph.library.function),
        'bytes': graph.ByteSize(), 'trace_count': evaluate.experimental_get_tracing_count()}
    if jit:
        hlo = evaluate.experimental_get_compiler_ir(operands)(stage='hlo')
        other = evaluate.experimental_get_compiler_ir(changed)(stage='hlo')
        graph_info.update(hlo_bytes=len(hlo.encode()), hlo_unchanged=stable_hlo(hlo) == stable_hlo(other))
    numerical_passed = all(row['passed'] for row in comparisons)
    report = {'schema': 'filter_srukf_svd_cost.v1', 'arm': arm, 'horizon': horizon,
        'baseline': BASELINE, 'original_source_sha256': sources, 'gpu': gpu, 'jit_compile': jit,
        'construction_trace_seconds': construction_trace_seconds, 'cold': cold, 'samples': samples,
        'changed': changed_cost, 'returned': returned_cost, 'snapshots': snapshots, 'program': graph_info,
        'comparisons': comparisons, 'numerical_passed': numerical_passed,
        'gpu_process_observation': sharing.payload(),
        'timing_scope': 'One reusable enclosing filter, including numerical output materialization; no public per-call construction.',
        'nonclaims': ['Inaccurate-arm costs cannot support a speed ranking.',
            'Fresh-process descriptive measurements; snapshots cannot establish exact native peak or eviction.',
            'Graph and CPU are explicit references; no nonlinear, analytical-score or posterior qualification.']}
    save(request, 'srukf-svd-cost.json', report)
    for name in initial:
        np.testing.assert_array_equal(returned[name], initial[name])
    assert graph_info['trace_count'] == 1 and (not jit or graph_info['hlo_unchanged'])
    if arm.startswith('after') or arm == 'prior_graph':
        assert numerical_passed, comparisons
