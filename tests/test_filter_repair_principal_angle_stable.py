"""Independent high-precision diagnostics for stable principal-angle evaluation."""

import ast
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import mpmath as mp
import numpy as np
import tensorflow as tf

from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_gaussian_binding_readback import runtime_receipt
from tests.test_filter_repair_principal_angle_precision import (
    FIXTURE,
    FIXTURE_SHA,
    memory,
    program,
    synchronize,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')
BASELINE = 'bf022dd90'
D = tf.float64


def save(request, name, value):
    directory = Path(request.config.getoption('xmlpath')).parent
    (directory/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def high_precision(first, second, rank):
    """80-digit eigensystem/SVD of the exact binary64 matrices, not rounded cosines."""
    with mp.workdps(80):
        first_mp, second_mp = mp.matrix(first), mp.matrix(second)
        va, qa = mp.eigsy(first_mp)
        vb, qb = mp.eigsy(second_mp)
        d = len(first)
        a, b = qa[:, d-rank:], qb[:, d-rank:]
        singular = mp.svd(a.T*b, compute_uv=False)
        angles = []
        for x in singular:
            if abs(1-x) <= mp.mpf('1e-12'):
                x = mp.mpf(1)
            angles.append(float(mp.acos(min(1, max(-1, x)))*180/mp.pi))
        gaps = [float((v[d-rank]-v[d-rank-1])/max(abs(v[0]), abs(v[d-1]))) if rank < d else None for v in (va, vb)]
        return {'angles': angles, 'relative_selected_eigengaps': gaps,
            'precision_digits': 80, 'source': 'exact stored binary64 matrices'}


def candidate_module(request):
    checkpoint = FrozenCheckpoint(BASELINE, 'stable_angle_reference')
    baseline_module = checkpoint.load('bayesfilter.inference.fixed_center_curvature')
    original = checkpoint.sources['bayesfilter/inference/fixed_center_curvature.py']
    node = next(n for n in ast.parse(original).body if isinstance(n, ast.FunctionDef)
                and n.name == '_precision_geometry_kernel')
    source = ast.get_source_segment(original, node)+'\n'
    old = '''    angles = tf.acos(tf.clip_by_value(singular, -1.0, 1.0)) * tf.constant(
        180.0 / math.pi, tf.float64
    )'''
    new = '''    first_basis = first_vectors * selected
    second_basis = second_vectors * selected
    residual = second_basis - tf.matmul(first_basis, overlap)
    sine = (xla_svd(residual, max_iter=100, epsilon=sys.float_info.epsilon,
        precision_config="").s if jit_compile else tf.linalg.svd(residual, compute_uv=False))
    paired_sine = tf.gather(sine, tf.maximum(rank - 1 - tf.range(dimension), 0))
    radians = tf.math.atan2(paired_sine, tf.clip_by_value(singular, 0.0, 1.0))
    angles = tf.where(singular == 1.0, tf.zeros_like(radians), radians) * tf.constant(
        180.0 / math.pi, tf.float64)
    angles = tf.where(tf.range(dimension) < rank, angles, tf.constant(90.0, tf.float64))'''
    assert source.count(old) == 1
    changed = source.replace(old, new)
    path = Path(request.config.getoption('xmlpath')).parent/'stable-angle-diagnostic.py'
    path.write_text(changed)
    namespace = dict(baseline_module.__dict__)
    exec(compile(changed, str(path), 'exec'), namespace)  # noqa: S102 - diagnostic source intervention
    class Module:
        _precision_geometry_kernel = staticmethod(namespace['_precision_geometry_kernel'])
    return Module, hashlib.sha256(source.encode()).hexdigest(), hashlib.sha256(changed.encode()).hexdigest(), baseline_module


def graph(owner, args, request, name):
    concrete = owner.get_concrete_function()
    definition = concrete.graph.as_graph_def()
    nodes = [*definition.node, *(n for f in definition.library.function for n in f.node_def)]
    assert concrete.function_def.attr['_XlaMustCompile'].b
    assert not any('/pfor/' in n.name for n in nodes)
    assert not {n.op for n in nodes} & {'PyFunc', 'EagerPyFunc', 'PyFuncStateless', 'XlaHostCompute'}
    assert owner.experimental_get_tracing_count() == 1
    hlo = owner.experimental_get_compiler_ir(*args)(stage='hlo')
    path = Path(request.config.getoption('xmlpath')).parent/f'{name}.hlo.txt'
    path.write_text(hlo)
    return {'trace_count': 1, 'jit_compile': True, 'no_host_callbacks_or_pfor': True,
            'hlo_sha256': hashlib.sha256(hlo.encode()).hexdigest()}


def test_stable_angle_candidate(request):
    tf.config.experimental.enable_tensor_float_32_execution(False)
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA
    fixture = json.loads(FIXTURE.read_text())
    first, second, rank = fixture['first_precision'], fixture['second_precision'], fixture['rank']
    authority = high_precision(first, second, rank)
    module, source_hash, candidate_hash, baseline_module = candidate_module(request)
    args = tf.constant(first, D), tf.constant(second, D), tf.constant(rank, tf.int32)
    results = {}
    for name, owner in (('baseline', program(baseline_module, len(first))), ('candidate', program(module, len(first)))):
        before = memory()
        started = time.perf_counter()
        output = synchronize(owner(*args))
        cold = time.perf_counter()-started
        timings = []
        for _ in range(10):
            started = time.perf_counter()
            replay = synchronize(owner(*args))
            timings.append(time.perf_counter()-started)
            for a, b in zip(output, replay, strict=True):
                np.testing.assert_array_equal(a, b)
        results[name] = {'outputs': [np.asarray(x).tolist() for x in output],
            'cold_seconds': cold, 'warm_seconds': timings, 'memory_before': before,
            'memory_after': memory(), 'graph': graph(owner, args, request, name)}
    for name, result in results.items():
        angles = np.asarray(result['outputs'][3][:rank])
        expected = np.asarray(authority['angles'])
        error = np.abs(angles-expected)
        bounds = 1e-10+1e-10*np.abs(expected)
        result.update(angle_error=error.tolist(), bounds=bounds.tolist(), accuracy_passed=bool(np.all(error<=bounds)))
    metrics_error = []
    for index in (0, 1, 2, 4, 5, 6):
        a, b = np.asarray(results['candidate']['outputs'][index]), np.asarray(results['baseline']['outputs'][index])
        metrics_error.append(float(np.max(np.abs(a-b))))
        np.testing.assert_allclose(a, b, atol=1e-10, rtol=1e-10)
    analytic_owner = program(module, 4)
    analytic = []
    for angle in (0., .00001, .0002, .005, 4.9, 5.1, 40., 89., 90.):
        radians = angle*math.pi/180
        q = np.eye(4)
        q[0, 0] = q[2, 2] = math.cos(radians)
        q[0, 2] = -math.sin(radians)
        q[2, 0] = math.sin(radians)
        p = np.diag([1., 2., 4., 8.])
        output = synchronize(analytic_owner(tf.constant(p, D), tf.constant(q@p@q.T, D), tf.constant(2)))
        expected = 0. if abs(1-math.cos(radians)) <= 1e-12 else angle
        observed = output[3][:2]
        np.testing.assert_allclose(observed, [0., expected], atol=1e-10, rtol=1e-10)
        assert (max(observed) <= 5.) == (expected <= 5.)
        analytic.append({'angle': angle, 'expected': expected, 'actual': observed.tolist(), 'five_degree_decision_preserved': True})
    for active_rank in (1, 3, 4):
        output = synchronize(analytic_owner(tf.constant(np.diag([1., 2., 4., 8.]), D),
            tf.constant(np.diag([1., 2., 4., 8.]), D), tf.constant(active_rank)))
        np.testing.assert_array_equal(output[3], [0.]*active_rank+[90.]*(4-active_rank))
        assert output[2] == active_rank
    # Repeated eigenvalues cut by a requested rank do not define a unique
    # selected eigenspace; record the limitation instead of demanding equality.
    repeated = high_precision(np.eye(4).tolist(), np.eye(4).tolist(), 2)
    assert repeated['relative_selected_eigengaps'] == [0., 0.]
    report = {'schema': 'filter_principal_angle_stable_candidate.v1', 'baseline': BASELINE,
        'source_sha256': hashlib.sha256(subprocess.check_output(['git', 'show', f'{BASELINE}:bayesfilter/inference/fixed_center_curvature.py'], cwd=ROOT)).hexdigest(),
        'function_source_sha256': source_hash, 'candidate_source_sha256': candidate_hash,
        'fixture_sha256': FIXTURE_SHA, 'authority': authority, 'results': results,
        'other_metrics_max_abs': metrics_error, 'analytic': analytic,
        'repeated_eigenvalue_case': {'relative_gaps': repeated['relative_selected_eigengaps'],
            'status': 'non_unique_selected_subspace_not_an_angle_arithmetic_failure'},
        'cost_scope': 'same-process baseline/candidate sequential screen; not matched fresh-process cost admission',
        'runtime_changed': False, 'device': 'CPU' if os.environ['CUDA_VISIBLE_DEVICES'] == '-1' else 'GPU'}
    save(request, 'stable-angle-candidate.json', report)
    assert results['candidate']['accuracy_passed'], results['candidate']['angle_error']
    assert min(authority['relative_selected_eigengaps']) > 100*sys.float_info.epsilon


def test_saved_stable_candidate(request):
    records = {}
    for path in sorted(RAW.glob('run-*/run.json')):
        if int(path.parent.name[4:]) <= 4913:
            continue
        m = json.loads(path.read_text())
        if m['key'][1] not in ('principal_angle_stable_probe_cpu', 'principal_angle_stable_probe_gpu'):
            continue
        assert m['state'] == 'passed' and m['test_evidence']['passed']
        r = json.loads((path.parent/'stable-angle-candidate.json').read_text())
        assert r['source_sha256'] == hashlib.sha256(subprocess.check_output(['git', 'show', f'{BASELINE}:bayesfilter/inference/fixed_center_curvature.py'], cwd=ROOT)).hexdigest()
        assert r['results']['candidate']['accuracy_passed']
        r['runtime_receipt'] = runtime_receipt(path.parent, m)
        assert r['device'] not in records
        records[r['device']] = r
    assert set(records) == {'CPU', 'GPU'}
    assert records['CPU']['authority'] == records['GPU']['authority']
    saved = RAW/'run-04617/factor-precision-symmetry-qualification.json'
    differences = json.loads(saved.read_text())['remaining_CPU_GPU_record_differences']
    classification = Counter()
    for item in differences:
        parts = item['path'].split('/')
        if 'fits' in parts:
            i = parts.index('fits')
            label = 'unselected_fit_'+parts[i+1]
        elif 'stability' in parts or 'selection' in parts:
            label = 'selection_or_stability'
        else:
            label = 'other'
        classification[label] += 1
    save(request, 'stable-angle-readback.json', {'records': records,
        'saved_full_record_difference_count': len(differences), 'classified_paths': dict(classification),
        'angle_leaf_count': sum('angle' in x['path'] for x in differences),
        'source_comparison_sha256': hashlib.sha256(saved.read_bytes()).hexdigest(),
        'nonclaims': ['No shared runtime change or full fitted-record closure from this diagnostic.',
            'Full fitted records, decisions and terminal costs remain required before broader admission.']})
