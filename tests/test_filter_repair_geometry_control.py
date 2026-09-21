"""Original-source full-record checks for internal quadratic control kernels."""

import dataclasses
import gc
import json
import math
import re
import weakref
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import quadratic_geometry as geometry
from bayesfilter.inference import quadratic_geometry_control_tf as runtime
from bayesfilter.inference.quadratic_geometry_control_report import (
    center_refinement_report,
    exact_replay_report,
)
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_posterior_curvature import (
    stable_hlo as posterior_stable_hlo,
)
from tests.test_filter_repair_quadratic_batches import _equal_records

D = tf.float64


def stable_hlo(hlo):
    # 02484 differs only in these Grappler dummy-source constant uniquifiers.
    return re.sub(r'(op_name="zeros(?:_\d+)?/)_\d+(" source_file="dummy_file_name")',
                  r'\1_ID\2', posterior_stable_hlo(hlo))


@lru_cache(maxsize=2)
def source(revision):
    checkpoint = FrozenCheckpoint(revision, 'geometry_control_original')
    return checkpoint, checkpoint.load('bayesfilter.inference.quadratic_geometry')


def clean(value):
    if isinstance(value, dict):
        return {name: clean(child) for name, child in value.items()}
    if tf.is_tensor(value) or isinstance(value, np.ndarray):
        return clean(np.asarray(value).tolist())
    if isinstance(value, (list, tuple)):
        return [clean(child) for child in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def save(request, name, record):
    directory = Path(request.config.getoption('xmlpath')).parent
    with (directory / name).open('x') as output:
        json.dump(clean(record), output, indent=2, allow_nan=False)
        output.write('\n')


def fixture(dimension, case, *, record_calls=True):
    precision = tf.linalg.diag(tf.cast(tf.range(dimension) + 2, D)) + .05
    scale = tf.cast(tf.range(dimension), D) * .1 + .8
    center = tf.cast(tf.range(dimension), D) * .01
    mode = tf.cast(tf.range(dimension), D) * .02 + .12
    delta = center - mode
    score = -tf.linalg.matvec(precision, delta)
    value = .5 * tf.reduce_sum(delta * score)
    linear = score * scale
    local_precision = scale[:, None] * precision * scale[None]
    cfg = geometry.LowRankSPDQuadraticGeometryConfig(
        constrain_center_refinement_to_trust_region=case != 'unconstrained',
        trust_radius=.03 if case == 'boundary' else 1.0)
    position = tf.Variable(tf.zeros([dimension], D)) if record_calls else None
    if record_calls:
        with tf.device(position.device):
            calls = tf.Variable(0, dtype=tf.int64)
        assert calls.device == position.device
    else:
        calls = None

    def callback(point):
        if calls is not None:
            calls.assign_add(1)
            position.assign(point)
        p = tf.linalg.diag(tf.cast(tf.range(dimension) + 2, D)) + .05
        difference = point - (tf.cast(tf.range(dimension), D) * .02 + .12)
        result_score = -tf.linalg.matvec(p, difference)
        result_value = .5 * tf.reduce_sum(difference * result_score)
        if case == 'nonfinite_value':
            result_value = tf.constant(float('nan'), D)
        if case == 'nonfinite_score':
            result_score = tf.fill([dimension], tf.constant(float('inf'), D))
        if case == 'decrease':
            result_value -= 5.0
            result_score += 3.0
        return result_value, result_score

    if case == 'zero':
        linear = tf.zeros_like(linear)
    elif case == 'nonspd':
        local_precision = -local_precision
    elif case == 'invalid':
        local_precision = tf.fill([dimension, dimension], tf.constant(float('nan'), D))
    args = (center, scale, local_precision, linear, value, tf.linalg.norm(score * scale))
    return callback, cfg, args, calls, position


def reference(module, callback, cfg, args):
    return module._evaluate_center_refinement(value_and_score_fn=callback,
        center=args[0].numpy(), scale=args[1].numpy(), precision=args[2].numpy(), linear=args[3].numpy(),
        cfg=module.LowRankSPDQuadraticGeometryConfig(**dataclasses.asdict(cfg)),
        center_value=float(args[4]), center_score_norm=float(args[5]))


@pytest.mark.parametrize('dimension', [1, 3, 5])
@pytest.mark.parametrize('case', ['interior', 'boundary', 'unconstrained', 'zero',
    'nonspd', 'invalid', 'nonfinite_value', 'nonfinite_score', 'decrease'])
def test_full_center_proposal_records_against_original(dimension, case, request):
    callback, cfg, args, calls, position = fixture(dimension, case)
    records, counts, positions, hashes = {}, {}, {}, {}
    for revision in ('3582b4ac', '813eed67'):
        checkpoint, module = source(revision)
        calls.assign(0)
        position.assign(tf.zeros_like(position))
        records[revision] = clean(reference(module, callback, cfg, args))
        counts[revision], positions[revision] = int(calls), position.numpy().tolist()
        hashes[revision] = checkpoint.hashes()
    for jit in (False, True):
        calls.assign(0)
        position.assign(tf.zeros_like(position))
        program = runtime.make_center_refinement_program(callback, dimension, cfg, jit_compile=jit)
        raw = program(*args)
        key = 'xla' if jit else 'graph'
        records[key] = clean(center_refinement_report(raw, cfg, args[4], args[5]))
        counts[key], positions[key] = int(calls), position.numpy().tolist()
        assert counts[key] == int(raw['target_called'])
    save(request, f'geometry-control-{case}-{dimension}.json',
        {'records': records, 'target_calls': counts, 'positions': positions, 'source_sha256': hashes})
    # The frozen checkpoint's raw-XLA eigenvectors have a demonstrated residual
    # error (02481/02482); retain its full record as attribution, not authority.
    for key in ('graph', 'xla'):
        _equal_records(records[key], records['3582b4ac'])
        assert counts[key] == counts['3582b4ac']
        np.testing.assert_allclose(positions[key], positions['3582b4ac'], atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize('constrained', [False, True])
def test_exact_singular_proposal_skips_target(constrained, request):
    callback, cfg, args, calls, _ = fixture(3, 'interior')
    cfg = dataclasses.replace(cfg, constrain_center_refinement_to_trust_region=constrained)
    args = (*args[:2], tf.zeros([3, 3], D), *args[3:])
    _, baseline = source('3582b4ac')
    expected = clean(reference(baseline, callback, cfg, args))
    assert int(calls) == 0
    actual = center_refinement_report(runtime.make_center_refinement_program(callback, 3, cfg)(*args), cfg, args[4], args[5])
    save(request, f'geometry-control-singular-{constrained}.json', {'original': expected, 'actual': actual})
    assert int(calls) == 0
    _equal_records(clean(actual), expected)


@pytest.mark.parametrize('case', ['missing', 'match', 'changed_value', 'changed_score', 'nan_value', 'nan_score'])
def test_complete_incumbent_replay_records(case, request):
    counter = tf.Variable(0, dtype=tf.int64)
    point = tf.constant([.3, -.2, .1], D)

    def callback(x):
        counter.assign_add(1)
        value = -.5 * tf.reduce_sum(x * x)
        score = -x
        return (tf.constant(float('nan'), D) if case == 'nan_value' else value,
                tf.fill([3], tf.constant(float('nan'), D)) if case == 'nan_score' else score)

    value = -.5 * tf.reduce_sum(point * point) + (1.0 if case == 'changed_value' else 0.0)
    score = -point + (1.0 if case == 'changed_score' else 0.0)
    outputs = {}
    for revision in ('3582b4ac', '813eed67'):
        checkpoint, module = source(revision)
        candidate_class = checkpoint.load('bayesfilter.inference._exact_incumbent').ExactCandidate
        incumbent = None if case == 'missing' else candidate_class(
            point.numpy(), float(value), score.numpy(), 2, 'design')
        counter.assign(0)
        outputs[revision] = clean(module._canonical_replay(callback, incumbent, evaluation_index=7))
        assert int(counter) == int(case != 'missing')
    for jit in (False, True):
        counter.assign(0)
        raw = runtime.make_exact_replay_program(callback, 3, jit_compile=jit)(point, value, score, case != 'missing')
        outputs[str(jit)] = clean(exact_replay_report(raw, evaluation_index=7, source='design'))
        assert int(counter) == int(case != 'missing')
    save(request, f'geometry-replay-{case}.json', outputs)
    for record in outputs.values():
        _equal_records(record, outputs['3582b4ac'])


def test_changed_inputs_are_runtime_operands_and_enclosing_xla(request):
    callback, cfg, args, _, _ = fixture(3, 'interior', record_calls=False)
    program = runtime.make_center_refinement_program(callback, 3, cfg)
    changed = (args[0] + .03, args[1] * 1.1, args[2] * 1.03, args[3] * .9, args[4] - .1, args[5] * 1.1)
    records = []
    for inputs in (args, changed):
        raw = program(*inputs)
        actual = clean(center_refinement_report(raw, cfg, inputs[4], inputs[5]))
        expected = clean(reference(source('3582b4ac')[1], callback, cfg, inputs))
        records.append({'actual': actual, 'original': expected})
        _equal_records(actual, expected)
    assert program.experimental_get_tracing_count() == 1
    hlo = program.experimental_get_compiler_ir(*args)(stage='hlo')
    changed_hlo = program.experimental_get_compiler_ir(*changed)(stage='hlo')
    directory = Path(request.config.getoption('xmlpath')).parent
    (directory / 'geometry-control-first.hlo').write_text(hlo)
    (directory / 'geometry-control-changed.hlo').write_text(changed_hlo)
    assert stable_hlo(hlo) == stable_hlo(changed_hlo)
    assert len(re.findall(r'\bparameter\((\d+)\)', hlo[hlo.rfind('\nENTRY '):])) == 6
    graph = program.get_concrete_function().graph.as_graph_def()
    assert not any(node.op in ('PyFunc', 'PyFuncStateless', 'EagerPyFunc')
        for nodes in (graph.node, *(function.node_def for function in graph.library.function)) for node in nodes)
    outer = tf.function(lambda *values: program(*values), input_signature=program.input_signature,
        autograph=False, jit_compile=True)
    _equal_records(clean(outer(*changed)), clean(program(*changed)))
    save(request, 'geometry-control-operands.json', {'records': records, 'trace_count': 1, 'runtime_operands': 6})


def test_cache_uses_callback_identity_and_releases_previous_graph():
    class EqualTarget:
        def __init__(self, shift):
            self.shift = tf.Variable(shift, dtype=D)

        def __eq__(self, other):
            return isinstance(other, EqualTarget)

        def __call__(self, point):
            delta = point - self.shift
            return -.5 * tf.reduce_sum(delta * delta), -delta

    runtime.clear_center_refinement_cache()
    _, cfg, args, _, _ = fixture(3, 'interior', record_calls=False)
    first = EqualTarget(.1)
    program = runtime.center_refinement_program(first, 3, cfg)
    before = program(*args)
    assert runtime.center_refinement_program(first, 3, dataclasses.replace(cfg, seed=456)) is program
    first.shift.assign(.3)
    after = program(*args)
    assert float(before['refined_value']) != float(after['refined_value'])
    assert program.experimental_get_tracing_count() == 1
    reference_callback, reference_graph = weakref.ref(first), weakref.ref(program.get_concrete_function().graph)
    second = EqualTarget(.2)
    replacement = runtime.center_refinement_program(second, 3, cfg)
    assert replacement is not program
    del first, program
    gc.collect()
    assert reference_callback() is None
    assert reference_graph() is None
    runtime.clear_center_refinement_cache()


@pytest.mark.parametrize('gate', ['radius', 'value', 'score', 'actual'])
@pytest.mark.parametrize('outside', [False, True])
def test_acceptance_boundaries_retain_original_decisions(gate, outside, request):
    cfg = geometry.LowRankSPDQuadraticGeometryConfig(trust_radius=1.,
        center_log_prob_tolerance=.125, center_score_improvement_factor=.5,
        constrain_center_refinement_to_trust_region=gate == 'actual')
    linear, center_value, target_value, target_score = .25, 0., 0., 0.
    if gate == 'radius':
        linear = float(np.nextafter(1., np.inf)) if outside else 1.
    elif gate == 'value':
        target_value = float(np.nextafter(-.125, -np.inf)) if outside else -.125
    elif gate == 'score':
        target_score = float(np.nextafter(.5, np.inf)) if outside else .5
    else:
        center_value = 1.
        target_value = 1. if outside else float(np.nextafter(1., np.inf))

    def callback(_point):
        return tf.constant(target_value, D), tf.constant([target_score], D)

    args = (tf.zeros([1], D), tf.ones([1], D), tf.eye(1, dtype=D),
        tf.constant([linear], D), tf.constant(center_value, D), tf.constant(1., D))
    expected = clean(reference(source('3582b4ac')[1], callback, cfg, args))
    results = {}
    for jit in (False, True):
        raw = runtime.make_center_refinement_program(callback, 1, cfg, jit_compile=jit)(*args)
        results[str(jit)] = clean(center_refinement_report(raw, cfg, args[4], args[5]))
    save(request, f'geometry-control-boundary-{gate}-{outside}.json', {'original': expected, 'results': results})
    assert expected['accepted'] is not outside
    for actual in results.values():
        _equal_records(actual, expected)


@pytest.mark.parametrize('design', ['duplicate', 'permuted', 'tiny'])
def test_unconstrained_zero_pivot_policy_matches_original(design, request):
    callback, cfg, args, calls, _ = fixture(3, 'unconstrained')
    matrix = {'duplicate': tf.ones([3, 3], D),
        'permuted': tf.constant([[0., 1., 0.], [1., 0., 0.], [0., 0., 2.]], D),
        'tiny': tf.linalg.diag(tf.constant([1e-20, 1., 2.], D))}[design]
    args = (*args[:2], matrix, *args[3:])
    expected = clean(reference(source('3582b4ac')[1], callback, cfg, args))
    expected_calls = int(calls)
    results = {}
    for jit in (False, True):
        calls.assign(0)
        raw = runtime.make_center_refinement_program(callback, 3, cfg, jit_compile=jit)(*args)
        results[str(jit)] = clean(center_refinement_report(raw, cfg, args[4], args[5]))
        assert int(calls) == expected_calls
    save(request, f'geometry-control-pivot-{design}.json', {'original': expected, 'results': results})
    for actual in results.values():
        _equal_records(actual, expected)


def test_trust_eigensystem_attribution(request):
    from tensorflow.compiler.tf2xla.ops.gen_xla_ops import xla_self_adjoint_eig

    from bayesfilter.inference.mass_matrix_tf import eigenpair_program

    _, _, args, _, _ = fixture(3, 'interior', record_calls=False)
    matrix, linear = args[2], args[3]

    @tf.function(input_signature=[tf.TensorSpec([3, 3], D)], autograph=False, jit_compile=True)
    def raw_eigenpairs(value):
        return xla_self_adjoint_eig(value, lower=True, max_iter=100, epsilon=math.ulp(1.0))

    results = {}
    expected = np.linalg.solve(matrix.numpy(), linear.numpy())
    for name, program in (('raw', raw_eigenpairs), ('refined', eigenpair_program(3))):
        values, vectors = program(matrix)
        solution = vectors.numpy() @ ((vectors.numpy().T @ linear.numpy()) / values.numpy())
        results[name] = {'eigenvalues': values.numpy().tolist(), 'eigenvectors': vectors.numpy().tolist(),
            'eigen_residual_norm': float(np.linalg.norm(matrix.numpy() @ vectors.numpy() - vectors.numpy() * values.numpy())),
            'orthogonality_residual_norm': float(np.linalg.norm(vectors.numpy().T @ vectors.numpy() - np.eye(3))),
            'solution': solution.tolist(), 'max_solution_error': float(np.max(np.abs(solution - expected)))}
    save(request, 'geometry-control-eigen-attribution.json',
        {'role': 'source_attribution_only_no_comparison_waiver', 'matrix': matrix, 'linear': linear,
         'independent_solution': expected, 'results': results})
    assert results['refined']['eigen_residual_norm'] < 1e-12
    np.testing.assert_allclose(results['refined']['solution'], expected, atol=1e-10, rtol=1e-10)


def test_scalar_graph_and_lu_attribution(request):
    callback, cfg, args, _, _ = fixture(1, 'interior', record_calls=False)
    outcomes = {}
    for jit in (False, True):
        program = runtime.make_center_refinement_program(callback, 1, cfg, jit_compile=jit)
        raw = program(*args)
        outcomes[f'controller_{jit}'] = {key: {'shape': value.shape.as_list(), 'dtype': value.dtype.name,
            'value': clean(value)} for key, value in raw.items()}
    signature = [tf.TensorSpec([1, 1], D), tf.TensorSpec([1], D), tf.TensorSpec([], D)]
    for jit in (False, True):
        def solve(matrix, linear, radius):
            return geometry._trust_region_kernel(matrix, linear, radius, eigenpairs=tf.linalg.eigh)
        try:
            raw = tf.function(solve, input_signature=signature, jit_compile=jit, autograph=False)(args[2], args[3], 1.)
            outcomes[f'trust_{jit}'] = {key: {'shape': value.shape.as_list(), 'dtype': value.dtype.name,
                'value': clean(value)} for key, value in raw.items()}
        except tf.errors.OpError as exc:
            outcomes[f'trust_{jit}'] = {'error': str(exc)}
    lu = tf.function(tf.linalg.lu, input_signature=[tf.TensorSpec([3, 3], D)], jit_compile=True, autograph=False)
    try:
        factors, perm = lu(tf.eye(3, dtype=D))
        outcomes['lu'] = {'factors': clean(factors), 'permutation': clean(perm)}
    except tf.errors.OpError as exc:
        outcomes['lu'] = {'error': str(exc)}
    save(request, 'geometry-control-scalar-and-lu.json', outcomes)
