"""Diagnostic full-record checks of the isolated real DZ5 initializer adapter.

NumPy reads only the independent archived comparator and saved NPZ files.
The candidate is the external adapter source, with real configuration parsing.
"""

import contextlib
import dataclasses
import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import dense_initializer_seeded_tf as seeded
from bayesfilter.inference import fixed_center_curvature as fixed
from bayesfilter.inference.joint_center import JointCenterLocatorConfig
from bayesfilter.inference.posterior_local_initializer import (
    PosteriorLocalInitializerConfig,
)
from tests.test_filter_repair_block_capture import stable_hlo
from tests.test_filter_repair_dense_controller import (
    normalize_result,
    original_initializer,
)
from tests.test_filter_repair_dense_validated_fit import _thresholds
from tests.test_filter_repair_geometry_control import clean, save
from tests.test_filter_repair_quadratic_batches import _equal_records

SNAPSHOT = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/'
    'filter-gradient-repair-20260917/dz5-initializer-adapter-20260928-r1')
MF = Path('/home/ubuntu/workspace/MacroFinance-dz5-neutra')
D = tf.float64


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_imports(manifest):
    """The small fixture uses current code only when it matches the snapshot."""
    loaded = {}
    forbidden = ('filters', 'inference.hmc', 'inference.mass_matrix', 'inference.posterior_adapter')
    assert not [name for name in sys.modules if any(name == prefix or name.startswith(prefix + '.')
        for prefix in forbidden)]
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if path and (name == 'bayesfilter' or name.startswith('bayesfilter.')):
            path = Path(path).resolve()
            relative = path.relative_to(Path(__file__).resolve().parents[1])
            canonical = '/home/ubuntu/workspace/BayesFilter/' + str(relative)
            digest = sha(path)
            assert manifest['sources'][canonical]['sha256'] == digest, canonical
            loaded[name] = {'path': str(path), 'sha256': digest}
    return loaded


def load_candidate(monkeypatch):
    manifest = json.loads((SNAPSHOT / 'manifest.json').read_text())
    tree = SNAPSHOT / str(MF).lstrip('/')
    monkeypatch.syspath_prepend(str(tree))
    for name in ('bayesfilter_estimation', 'bayesfilter_estimation_runtime'):
        module = importlib.import_module(name)
        assert Path(module.__file__).resolve().is_relative_to(tree)
        assert sha(Path(module.__file__)) == manifest['sources'][str(MF / f'{name}.py')]['sha256']
    path = tree / 'bayesfilter_estimation_initialization.py'
    assert sha(path) == manifest['adapter_source_sha256']
    spec = importlib.util.spec_from_file_location('dz5_initializer_adapter_candidate', path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    audit_imports(manifest)
    return module.initialize_dense_local, manifest


def read_archive(root):
    path = root / 'initializer_evaluations.npz'
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


@pytest.mark.parametrize('dimension,case', [(1, 'healthy'), (3, 'healthy'),
    (3, 'invalid_locator'), (3, 'invalid_cloud')])
def test_real_initializer_adapter_records(dimension, case, tmp_path, request, monkeypatch):
    candidate, manifest = load_candidate(monkeypatch)
    rows = [3 * dimension] * 2 + [2 * dimension] * 2 + [2 * dimension + 1]
    configuration = PosteriorLocalInitializerConfig(
        locator_config=JointCenterLocatorConfig(max_iterations=4,
            max_objective_evaluations=40, gradient_tolerance=1e-8),
        max_curvature_attempts=2, training_rows_per_replicate=rows[0],
        selection_rows_per_replicate=rows[2], audit_rows=rows[-1],
        replicate_count=2, seed=(215, 91), curvature_radius=.13)
    thresholds = _thresholds(fixed)
    reference, reference_provenance = original_initializer(configuration, thresholds)
    covariance = tf.constant([[1. / 1.7]], D)
    if dimension == 3:
        loadings = tf.constant([.2, .35, -.15], D)
        marginal = tf.constant([.9, 1.2, 1.4], D)
        covariance = marginal[:, None] * (tf.linalg.diag(1. - loadings**2)
            + loadings[:, None] * loadings[None, :]) * marginal[None, :]
    precision = tf.linalg.inv(covariance)
    calls = tf.Variable(0, dtype=tf.int64, trainable=False)
    count = tf.Variable(0, dtype=tf.int64, trainable=False)
    positions = tf.Variable(tf.zeros([1000, dimension], D), trainable=False)
    extents = tf.Variable(tf.zeros([1000], tf.int64), trainable=False)

    def atomic(points):
        n = points.shape[0]
        index = count.assign_add(n) - n
        call = calls.assign_add(1) - 1
        positions.scatter_nd_update((index + tf.range(n, dtype=tf.int64))[:, None], points)
        extents.scatter_nd_update([[call]], [tf.cast(n, tf.int64)])
        score = -tf.linalg.matmul(points, precision, transpose_b=True)
        value = .5 * tf.reduce_sum(points * score, axis=1)
        valid = tf.ones([n], tf.bool)
        if case == 'invalid_locator' or (case == 'invalid_cloud' and n != 1):
            valid = tf.zeros_like(valid)
        return value, score, valid

    def forbidden_scalar_or_status(*_):
        raise AssertionError('Atomic target must not fall back to separate value/status calls')

    model = SimpleNamespace(training_target=SimpleNamespace(parameter_dim=dimension,
        batch_value_score_and_validity=atomic, batch_value_and_score=forbidden_scalar_or_status,
        target_status_telemetry=forbidden_scalar_or_status),
        coordinate_scale=.8 + .2 * tf.cast(tf.range(dimension), D),
        initial_position=tf.zeros([dimension], D))
    recipe = SimpleNamespace(initializer=dataclasses.asdict(configuration), movement={},
        curvature_thresholds=dataclasses.asdict(thresholds), dense_center_score_max=1e-6)
    outputs, observations, archives, boundaries, owners = {}, {}, {}, {}, []
    original_type = seeded.SeededDenseInitializerProgram

    class ObservedProgram(original_type):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            owners.append(self)

    monkeypatch.setattr(seeded, 'SeededDenseInitializerProgram', ObservedProgram)

    for label, endpoint in (('reference', reference), ('candidate', candidate)):
        root = tmp_path / label
        root.mkdir()
        events = []

        @contextlib.contextmanager
        def boundary(name, events=events):
            events.append(('start', name))
            yield
            events.append(('end', name))

        context = SimpleNamespace(root=root, boundary=boundary)
        calls.assign(0)
        count.assign(0)
        outputs[label] = endpoint(model, recipe, context)
        observations[label] = clean({'calls': calls, 'rows': count,
            'positions': positions[:int(count)], 'extents': extents[:int(calls)]})
        archives[label] = read_archive(root)
        boundaries[label] = events
        for name, digest in outputs[label].get('artifacts', {}).items():
            assert sha(Path(name)) == digest
    assert len(owners) == 1
    owner = owners[0]
    concrete = owner.compiled.get_concrete_function()
    graph = concrete.graph.as_graph_def()
    nodes = [*graph.node, *(node for function in graph.library.function for node in function.node_def)]
    operands = (model.initial_position, model.coordinate_scale,
        tf.constant(configuration.seed, tf.int32), tf.constant(configuration.curvature_radius, D))
    changed = (operands[0] + tf.constant(.00001, D), operands[1] + tf.constant(.00001, D), *operands[2:])
    hlo = owner.compiled.experimental_get_compiler_ir(*operands)(stage='hlo')
    changed_hlo = owner.compiled.experimental_get_compiler_ir(*changed)(stage='hlo')
    report = {'schema': 'filter_dz5_initializer_adapter.v1', 'dimension': dimension, 'case': case,
        'loaded_bayesfilter_modules': audit_imports(manifest),
        'role': 'isolated_real_adapter_with_independent_Gaussian_fixture_not_actual_DZ5_target',
        'snapshot_manifest_sha256': sha(SNAPSHOT / 'manifest.json'),
        'adapter_sha256': manifest['adapter_source_sha256'], 'reference': reference_provenance,
        'outputs': clean(outputs), 'archives': clean(archives), 'callback_observations': observations,
        'boundaries': boundaries, 'jit_compile': bool(owner.compiled.function_spec.jit_compile),
        'trace_count': owner.compiled.experimental_get_tracing_count(),
        'input_signature': str(owner.compiled.input_signature),
        'hlo_unchanged': stable_hlo(hlo) == stable_hlo(changed_hlo),
        'hlo_sha256': hashlib.sha256(hlo.encode()).hexdigest(),
        'host_callback_ops': sorted({'PyFunc', 'EagerPyFunc', 'PyFuncStateless'} & {n.op for n in nodes}),
        'independent_covariance_theta': clean(covariance),
        'nonclaims': ['No CDF target/admission, process-lifetime bound, HMC, training or posterior qualification.']}
    save(request, 'dz5-initializer-adapter.json', report)
    _equal_records(normalize_result(outputs['candidate']), normalize_result(outputs['reference']))
    _equal_records(observations['candidate'], observations['reference'])
    _equal_records(clean(archives['candidate']), clean(archives['reference']))
    assert report['jit_compile'] and report['trace_count'] == 1
    assert report['hlo_unchanged'] and not report['host_callback_ops']
    assert boundaries['candidate'] == [('start', 'dense_initializer'), ('end', 'dense_initializer')]
    if case == 'healthy':
        assert outputs['candidate']['passed'] is True
        tf.debugging.assert_near(tf.constant(outputs['candidate']['initial_output_shift'], D),
            tf.zeros([dimension], D), atol=1e-7, rtol=1e-7)
        expected_scale_log = .5 * tf.math.log(tf.linalg.diag_part(covariance))
        tf.debugging.assert_near(tf.constant(outputs['candidate']['initial_output_scale_log'], D),
            expected_scale_log, atol=1e-7, rtol=1e-7)
    else:
        assert outputs['candidate']['passed'] is False
        assert outputs['candidate']['initial_output_shift'] is None
        assert outputs['candidate']['initial_output_scale_log'] is None
        if case == 'invalid_locator':
            assert not archives['candidate']
