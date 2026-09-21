"""Original initializer suffix with explicitly frozen basis, cloud and split."""

import dataclasses
import gc
import re
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import quadratic_geometry as geometry
from bayesfilter.inference.quadratic_geometry_fit_report import geometry_fit_report
from bayesfilter.inference.quadratic_geometry_fit_tf import (
    clear_geometry_fit_cache,
    geometry_fit_program,
    make_geometry_fit_program,
)
from tests.test_filter_repair_geometry_control import clean, save, source, stable_hlo
from tests.test_filter_repair_quadratic_batches import _equal_records

D = tf.float64
TOP_FIELDS = ('accepted', 'status', 'best_evaluated_position', 'best_evaluated_value',
    'best_evaluated_score', 'best_evaluated_index', 'best_evaluated_source', 'exact_evaluation_count',
    'center_refinement_accepted', 'refined_center', 'precision', 'covariance', 'q_basis',
    'linear_term', 'intercept', 'lambda0', 'mu')
DIAGNOSTIC_FIELDS = ('fit', 'train_rmse', 'holdout_count', 'holdout_rmse', 'holdout_threshold',
    'holdout_passed', 'precision_eigen_summary', 'covariance_eigen_summary',
    'center_refinement', 'best_evaluated_replay')


def fixture(dimension, case, *, counter=True, shift=0.):
    rank, train_rows = max(0, dimension - 1), 24
    holdout_rows = 0 if case == 'no_holdout' else 8
    total = train_rows + holdout_rows
    cfg = geometry.LowRankSPDQuadraticGeometryConfig(rank=max(1, rank), sample_count=total,
        min_samples_per_parameter=1, holdout_fraction=holdout_rows / total,
        trust_radius=1., constrain_center_refinement_to_trust_region=case != 'unconstrained',
        holdout_rmse_abs_tolerance=1e-14 if case == 'reject_holdout' else .05,
        holdout_rmse_rel_tolerance=1e-14 if case == 'reject_holdout' else .1)
    center = tf.cast(tf.range(dimension), D) * .02 + shift
    scale = (tf.cast(tf.range(dimension), D) * .1 + .8) * (1. + shift)
    basis = tf.eye(dimension, dtype=D)[:, :rank]
    offsets = tf.constant(np.random.default_rng(713 + dimension).normal(size=(total, dimension)) * .2, D)
    if case == 'zero_design':
        offsets = tf.zeros_like(offsets)
    calls = None
    if counter:
        with tf.device(center.device):
            calls = tf.Variable(0, dtype=tf.int64)

    def target(point):
        call = calls.assign_add(1) if calls is not None else tf.constant(0, tf.int64)
        delta = point - .13
        diagonal = tf.cast(dimension + 1 - tf.range(dimension), D)
        if case == 'constant':
            value, score = tf.constant(0., D), tf.zeros_like(point)
        elif case == 'flat':
            value, score = tf.reduce_sum(point), tf.ones_like(point)
        else:
            score = -diagonal * delta
            value = .5 * tf.reduce_sum(delta * score)
        if case in ('nonquadratic', 'reject_holdout'):
            value -= .05 * tf.reduce_sum(delta ** 4)
            score -= .2 * delta ** 3
        if case == 'indefinite':
            value, score = -value, -score
        if case == 'nonfinite_proposal':
            score = tf.where(call == total + 2, tf.fill([dimension], tf.constant(float('nan'), D)), score)
        if case == 'nonfinite_replay':
            value = tf.where(call == total + 3, tf.constant(float('nan'), D), value)
        if case == 'changed_replay':
            value += tf.where(call == total + 3, tf.constant(1., D), tf.constant(0., D))
        return value, score

    center_value, center_score = target(center)
    points = center[None] + offsets * scale[None]
    evaluated = [target(point) for point in points]
    values, scores = tf.stack([row[0] for row in evaluated]), tf.stack([row[1] for row in evaluated])
    # Independent earliest finite maximum over the exact prefix.
    best = int(np.argmax(np.r_[float(center_value), values.numpy()]))
    position = center if best == 0 else points[best - 1]
    value = center_value if best == 0 else values[best - 1]
    score = center_score if best == 0 else scores[best - 1]
    args = (center, scale, basis, offsets[holdout_rows:], values[holdout_rows:],
        scores[holdout_rows:] * scale, offsets[:holdout_rows], values[:holdout_rows],
        center_value, center_score * scale, position, value, score,
        tf.constant(best, tf.int64), tf.constant(0 if best == 0 else 2), tf.constant(True),
        tf.constant(total + 1, tf.int64))
    if calls is not None:
        calls.assign(0)
    return target, cfg, args, calls


def original_result(target, cfg, args, monkeypatch):
    checkpoint, module = source('3582b4ac')
    cloud = np.concatenate((args[6].numpy(), args[3].numpy()))
    basis = args[2].numpy()

    class FrozenRandomNumpy:
        random = SimpleNamespace(default_rng=lambda seed: SimpleNamespace(permutation=np.arange))

        def __getattr__(self, name):
            return getattr(np, name)

    with monkeypatch.context() as patch:
        patch.setattr(module, 'np', FrozenRandomNumpy())
        patch.setattr(module, '_sample_trust_ball', lambda *a, **kw: cloud.copy())
        patch.setattr(module, '_pilot_q_basis', lambda *a, **kw: (basis.copy(), {'frozen_basis': True}, ()))
        result = module.fit_low_rank_spd_quadratic_geometry(target, args[0].numpy(), scale=args[1].numpy(),
            config=module.LowRankSPDQuadraticGeometryConfig(**dataclasses.asdict(cfg)))
    fields = {name: getattr(result, name) for name in TOP_FIELDS}
    fields.update({name: result.diagnostics[name] for name in DIAGNOSTIC_FIELDS if name in result.diagnostics})
    return clean(fields), clean(result.payload(include_arrays=True)), checkpoint.hashes()


def materialize(raw, cfg, args):
    return clean(geometry_fit_report(raw, cfg, args[8], tf.linalg.norm(args[9]), holdout_rows=args[6].shape[0]))


@pytest.mark.parametrize('dimension', [1, 3, 5])
@pytest.mark.parametrize('case', ['gaussian', 'unconstrained', 'no_holdout', 'nonquadratic',
    'reject_holdout', 'zero_design', 'flat', 'indefinite', 'nonfinite_proposal', 'nonfinite_replay', 'changed_replay'])
def test_full_original_geometry_fit_suffix(dimension, case, monkeypatch, request):
    target, cfg, args, calls = fixture(dimension, case)
    expected, original_payload, hashes = original_result(target, cfg, args, monkeypatch)
    expected_calls = int(calls)
    records, counts = {}, {}
    for jit in (False, True):
        calls.assign(args[-1])
        try:
            program = make_geometry_fit_program(target, dimension, args[2].shape[1], args[3].shape[0],
                args[6].shape[0], cfg, jit_compile=jit)
            raw = program(*args)
            records[str(jit)] = materialize(raw, cfg, args)
            counts[str(jit)] = int(calls)
        except Exception as error:  # noqa: BLE001 - archive the failing candidate before the exact assertion.
            records[str(jit)] = {'error': type(error).__name__, 'message': str(error)}
            counts[str(jit)] = int(calls)
    save(request, f'geometry-fit-{case}-{dimension}.json', {
        'original': expected, 'complete_original_payload': original_payload, 'candidate': records,
        'original_calls': expected_calls, 'candidate_calls': counts, 'original_source_sha256': hashes,
        'inputs': args, 'frozen_reference_scope': 'pilot basis/cloud/permutation replaced with exact candidate inputs'})
    for jit in (False, True):
        _equal_records(records[str(jit)], expected)
        assert counts[str(jit)] == expected_calls
        assert records[str(jit)]['exact_evaluation_count'] == expected_calls
    if case == 'flat':
        assert not expected['center_refinement']['accepted']
        assert expected['best_evaluated_source'] == 'surrogate_replay'
    if case in ('nonfinite_replay', 'changed_replay'):
        assert not expected['best_evaluated_replay']['matches']


def test_nonfinite_fit_skips_target_and_preserves_original_fit_status(request):
    target, cfg, args, calls = fixture(3, 'gaussian')
    args = (*args[:4], tf.fill(args[4].shape, tf.constant(float('nan'), D)), *args[5:])
    checkpoint, module = source('3582b4ac')
    expected = module._fit_constrained_quadratic(args[3].numpy(), args[4].numpy(), args[5].numpy(),
        q_basis=args[2].numpy(), cfg=module.LowRankSPDQuadraticGeometryConfig(**dataclasses.asdict(cfg)),
        dim=3, rank=2, center_score_z=args[9].numpy())
    raw = make_geometry_fit_program(target, 3, 2, 24, 8, cfg)(*args)
    actual = materialize(raw, cfg, args)
    save(request, 'geometry-fit-invalid.json', {'original_fit': expected, 'candidate': actual,
        'original_source_sha256': checkpoint.hashes(), 'callback_calls': int(calls)})
    assert actual['fit'] == expected == {'status': 'fit_nonfinite'}
    assert not actual['accepted']
    assert int(calls) == 0
    assert actual['exact_evaluation_count'] == int(args[-1])


def test_fit_inputs_remain_runtime_operands(monkeypatch, request):
    target, cfg, args, _ = fixture(3, 'gaussian', counter=False)
    _, _, changed, _ = fixture(3, 'gaussian', counter=False, shift=.07)
    program = make_geometry_fit_program(target, 3, 2, 24, 8, cfg)
    results = [materialize(program(*inputs), cfg, inputs) for inputs in (args, changed)]
    expected, _, _ = original_result(target, cfg, args, monkeypatch)
    _equal_records(results[0], expected)
    changed_expected, _, _ = original_result(target, cfg, changed, monkeypatch)
    _equal_records(results[1], changed_expected)
    assert results[0] != results[1]
    assert program.experimental_get_tracing_count() == 1
    hlo = program.experimental_get_compiler_ir(*args)(stage='hlo')
    changed_hlo = program.experimental_get_compiler_ir(*changed)(stage='hlo')
    assert stable_hlo(hlo) == stable_hlo(changed_hlo)
    assert len(re.findall(r'\bparameter\((\d+)\)', hlo[hlo.rfind('\nENTRY '):])) == len(args)
    graph = program.get_concrete_function().graph.as_graph_def()
    assert not any(node.op in ('PyFunc', 'PyFuncStateless', 'EagerPyFunc')
        for nodes in (graph.node, *(fn.node_def for fn in graph.library.function)) for node in nodes)
    outer = tf.function(lambda *values: program(*values), input_signature=program.input_signature,
        autograph=False, jit_compile=True)
    _equal_records(materialize(outer(*changed), cfg, changed), results[1])
    save(request, 'geometry-fit-inputs.json', {'records': results, 'trace_count': 1, 'runtime_operands': len(args)})


@pytest.mark.parametrize('dimension', [1, 3, 5])
def test_exact_ties_keep_original_center(dimension, monkeypatch, request):
    target, cfg, args, calls = fixture(dimension, 'constant')
    expected, original_payload, hashes = original_result(target, cfg, args, monkeypatch)
    calls.assign(args[-1])
    program = make_geometry_fit_program(target, dimension, args[2].shape[1], 24, 8, cfg)
    actual = materialize(program(*args), cfg, args)
    save(request, f'geometry-fit-tie-{dimension}.json', {'original': expected, 'actual': actual,
        'original_payload': original_payload, 'original_source_sha256': hashes})
    _equal_records(actual, expected)
    assert actual['best_evaluated_source'] == 'center'
    assert actual['best_evaluated_index'] == 0


def scaled_target(base, strength):
    def target(point):
        value, score = base(point)
        return strength * value, strength * score
    return target


def test_fit_target_changes_and_resource_ownership(monkeypatch, request):
    base, cfg, args, _ = fixture(4, 'gaussian', counter=False)
    strength = tf.Variable(1., dtype=D)
    target = scaled_target(base, strength)

    program = make_geometry_fit_program(target, 4, 3, 24, 8, cfg)
    records = []
    for factor in (1., 2.):
        strength.assign(factor)
        inputs = (*args[:4], args[4] * factor, args[5] * factor, args[6], args[7] * factor,
            args[8] * factor, args[9] * factor, args[10], args[11] * factor, args[12] * factor, *args[13:])
        expected, _, _ = original_result(target, cfg, inputs, monkeypatch)
        actual = materialize(program(*inputs), cfg, inputs)
        records.append({'original': expected, 'actual': actual})
        _equal_records(actual, expected)
    assert program.experimental_get_tracing_count() == 1
    graph = program.get_concrete_function().graph
    watched = {name: weakref.ref(value) for name, value in (
        ('graph', graph), ('target', target), ('strength', strength))}
    del graph, program, target, strength
    gc.collect()
    released = {name: reference() is None for name, reference in watched.items()}
    save(request, 'geometry-fit-resource-ownership.json', {'records': records, 'released': released,
        'nonclaim': 'Python release is not proof of native executable eviction.'})
    assert all(released.values())


def test_fit_cache_binds_identity_numerical_settings_and_releases_old_target(request):
    class EqualTarget:
        def __init__(self, base):
            self.base = base

        def __eq__(self, other):
            return isinstance(other, EqualTarget)

        def __call__(self, point):
            return self.base(point)

    clear_geometry_fit_cache()
    base, cfg, args, _ = fixture(3, 'gaussian', counter=False)
    target = EqualTarget(base)
    program = geometry_fit_program(target, 3, 2, 24, 8, cfg)
    metadata_only = dataclasses.replace(cfg, seed=47, sample_count=77, fit_max_iterations=500)
    assert geometry_fit_program(target, 3, 2, 24, 8, metadata_only) is program
    first = materialize(program(*args), cfg, args)
    assert program.experimental_get_tracing_count() == 1
    graph = program.get_concrete_function().graph
    watched = (weakref.ref(target), weakref.ref(graph))
    replacement_target = EqualTarget(base)
    assert target == replacement_target
    replacement = geometry_fit_program(replacement_target, 3, 2, 24, 8, cfg)
    assert replacement is not program
    _equal_records(materialize(replacement(*args), cfg, args), first)
    del target, program, graph
    gc.collect()
    assert all(reference() is None for reference in watched)
    settings_changed = geometry_fit_program(replacement_target, 3, 2, 24, 8,
        dataclasses.replace(cfg, trust_radius=.75))
    assert settings_changed is not replacement
    clear_geometry_fit_cache()
    save(request, 'geometry-fit-cache.json', {'capacity': 1, 'identity_comparison': True,
        'metadata_only_reuses': True, 'numerical_setting_rebinds': True,
        'old_graph_and_target_released': True, 'full_result': first})
