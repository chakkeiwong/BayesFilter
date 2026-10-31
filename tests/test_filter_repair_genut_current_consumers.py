"""Live diagnostic call-chain witnesses; no canonical or numerical admission.

Frozen NumPy test fixtures are independent test data only. Resource counters
must preserve ordinary endpoint outputs before their branch counts are usable.
"""

import hashlib
import importlib
import os
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import genut_guided_proposal_tf as reset
from bayesfilter.highdim import ledh_unified_correction_tf as correction
from bayesfilter.highdim.ledh_alg1_contract import ENTRY_POINTS
from tests.highdim.test_ledh_canonical_score_full import _model
from tests.test_filter_repair_geometry_control import clean, save
from tests.test_filter_repair_ledh_seeded_public import _fixture
from tests.test_filter_repair_ledh_value_native import authorities as _authorities

ROOT = Path(__file__).resolve().parents[1]
RAW = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')
BASELINE = '9d83dc847'
DEPENDENCIES = tuple(f'bayesfilter/highdim/{name}.py' for name in (
    'dual_cap_genut_primal_tf', 'genut_guided_proposal_tf', 'ledh_canonical_filter_tf',
    'ledh_canonical_value_program_tf', 'ledh_canonical_score_tf', 'ledh_unified_correction_tf',
    'higher_moment_contract_e', 'ledh_alg1_contract'))


@pytest.fixture(scope='module')
def authorities():
    yield from _authorities.__wrapped__()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counted(original, counter):
    def call(*args, **kwargs):
        result = original(*args, **kwargs)
        operation = counter.assign_add(1)
        with tf.control_dependencies([operation]):
            return tf.nest.map_structure(tf.identity, result)
    return call


def compare(actual, expected):
    assert actual.keys() == expected.keys()
    errors = {}
    for key in actual:
        a, b = actual[key].numpy(), expected[key].numpy()
        if actual[key].dtype.is_floating:
            np.testing.assert_allclose(a, b, atol=1e-10, rtol=1e-10, equal_nan=True, err_msg=key)
            finite = np.isfinite(a) & np.isfinite(b)
            errors[key] = float(np.max(np.abs(a-b)[finite], initial=0.))
        else:
            np.testing.assert_array_equal(a, b, err_msg=key)
    return errors


def without_strings(record):
    return {k: (v.numpy().decode() if v.dtype == tf.string else clean(v)) for k, v in record.items()}


def test_registered_value_branch_wiring(authorities, monkeypatch, request):
    tf.config.experimental.enable_tensor_float_32_execution(False)
    registration = next(e for e in ENTRY_POINTS if e.callable_name == 'canonical_value_and_diagnostics')
    endpoint = getattr(importlib.import_module(registration.module), registration.callable_name)
    _, callbacks, observations, controls, _ = _fixture(authorities, 'one_step')
    reduced, trust = reset.dual_cap_genut_primal, reset.higher_moment_shape_jvp
    with tf.device(observations.device):
        reduced_counter = tf.Variable(0, dtype=tf.int64, trainable=False)
        trust_counter = tf.Variable(0, dtype=tf.int64, trainable=False)
    rows = []
    for branch, dual, full in (('disabled', False, False), ('reduced', True, False), ('trust', True, True)):
        options = {**controls, 'dual_cap_enabled': dual, 'trust_region_enabled': full}
        monkeypatch.setattr(reset, 'dual_cap_genut_primal', reduced)
        monkeypatch.setattr(reset, 'higher_moment_shape_jvp', trust)
        ordinary = endpoint(callbacks, observations, particle_count=8, seed=123, resample_seed=17, **options)
        reduced_counter.assign(0)
        trust_counter.assign(0)
        monkeypatch.setattr(reset, 'dual_cap_genut_primal', counted(reduced, reduced_counter))
        monkeypatch.setattr(reset, 'higher_moment_shape_jvp', counted(trust, trust_counter))
        observed = endpoint(callbacks, observations, particle_count=8, seed=123, resample_seed=17, **options)
        errors = compare(observed, ordinary)
        actual_counts = [int(reduced_counter), int(trust_counter)]
        expected_counts = {'disabled': [0, 0], 'reduced': [1, 0], 'trust': [0, 1]}[branch]
        assert actual_counts == expected_counts
        rows.append({'branch': branch, 'controls': options, 'runtime_counts': actual_counts,
            'expected_counts': expected_counts, 'maximum_output_error': max(errors.values()),
            'ordinary': without_strings(ordinary), 'observed': without_strings(observed)})
    save(request, 'genut-consumer-value.json', {'baseline': BASELINE,
        'registration': {'module': registration.module, 'callable': registration.callable_name},
        'device': 'CPU' if os.environ['CUDA_VISIBLE_DEVICES'] == '-1' else 'GPU',
        'sources': {path: digest(ROOT/path) for path in DEPENDENCIES}, 'records': rows,
        'nonclaims': ['Actual branch execution and preserved output do not qualify a refused reset.',
            'Reduced correction remains noncanonical; its saved gradient/cap findings stay open.']})


def test_registered_analytical_score_authority(monkeypatch, request):
    tf.config.experimental.enable_tensor_float_32_execution(False)
    registration = next(e for e in ENTRY_POINTS if e.callable_name == 'canonical_value_and_analytical_score')
    endpoint = getattr(importlib.import_module(registration.module), registration.callable_name)
    rng = np.random.default_rng(71)
    n, d, horizon = 16, 2, 2
    initial = tf.constant(rng.standard_normal((n, d)), tf.float64)
    covariances = tf.constant(np.stack([np.eye(d)]*n), tf.float64)
    noises = tf.constant(rng.standard_normal((horizon, n, d)), tf.float64)
    observations = tf.constant(rng.standard_normal((horizon, d)), tf.float64)
    design = tf.constant(np.tile(np.concatenate([np.eye(d), -np.eye(d)]), (n//(2*d), 1)), tf.float64)
    theta = tf.constant([0.6], tf.float64)
    options = {'flow_substeps': 8, 'reset_policy': 'contract_e', 'reset_design': design,
        'reset_sinkhorn_steps': 4, 'reset_balance_steps': 2, 'correction_steps': 2,
        'pairwise_steps': 1, 'coordinate_cap': .98}

    def owner():
        @tf.function(input_signature=[tf.TensorSpec([1], tf.float64),
            tf.TensorSpec([n, d], tf.float64), tf.TensorSpec([n, d, d], tf.float64),
            tf.TensorSpec([horizon, n, d], tf.float64), tf.TensorSpec([horizon, d], tf.float64)],
            jit_compile=True, autograph=False)
        def execute(t, x, c, e, y):
            value, score = endpoint(_model(), t, x, c, e, y, with_score=True, **options)
            return {'value': value, 'score': score}
        return execute

    args = (theta, initial, covariances, noises, observations)
    ordinary_owner = owner()
    ordinary = ordinary_owner(*args)
    original = correction.batched_higher_moment_shape_jvp
    with tf.device(initial.device):
        counter = tf.Variable(0, dtype=tf.int64, trainable=False)
    monkeypatch.setattr(correction, 'batched_higher_moment_shape_jvp', counted(original, counter))

    def forbidden(*args, **kwargs):
        raise AssertionError('analytical score must not call the reduced primal correction')

    monkeypatch.setattr(reset, 'dual_cap_genut_primal', forbidden)
    observed_owner = owner()
    observed = observed_owner(*args)
    errors = compare(observed, ordinary)
    assert int(counter) == horizon
    assert ordinary_owner.experimental_get_tracing_count() == observed_owner.experimental_get_tracing_count() == 1
    graph = observed_owner.get_concrete_function().graph.as_graph_def()
    operations = {node.op for node in graph.node}
    operations.update(node.op for function in graph.library.function for node in function.node_def)
    assert not operations & {'PyFunc', 'EagerPyFunc', 'PyFuncStateless', 'XlaHostCompute'}
    hlo = observed_owner.experimental_get_compiler_ir(*args)(stage='hlo')
    save(request, 'genut-consumer-analytical.json', {'baseline': BASELINE,
        'registration': {'module': registration.module, 'callable': registration.callable_name},
        'device': 'CPU' if os.environ['CUDA_VISIBLE_DEVICES'] == '-1' else 'GPU',
        'sources': {path: digest(ROOT/path) for path in DEPENDENCIES},
        'runtime_batched_jvp_calls': int(counter), 'reduced_primal_blocked': True,
        'ordinary': clean(ordinary), 'observed': clean(observed),
        'maximum_output_error': max(errors.values()),
        'finite': bool(tf.math.is_finite(ordinary['value']) & tf.reduce_all(tf.math.is_finite(ordinary['score']))),
        'hlo_sha256': hashlib.sha256(hlo.encode()).hexdigest(), 'jit_compile': True,
        'trace_count': 1,
        'nonclaims': ['This identifies the executed analytical correction authority; it is not a derivative accuracy or canonical conformance test.',
            'Saved reduced-primal reverse-gradient failures do not describe this different callable.']})
