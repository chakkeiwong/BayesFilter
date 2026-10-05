"""Original-source numerical authorities for fixed-capacity preparation."""

import ast
import hashlib
import inspect
import math
import random
import re
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.quadratic_geometry_prepare_tf import (
    _subnormal_square_units,
    make_direction_preparation_program,
    make_geometry_design_program,
    make_geometry_partition_program,
)
from tests.test_filter_repair_geometry_control import clean, save, source, stable_hlo
from tests.test_filter_repair_quadratic_batches import _equal_records

D = tf.float64
PARTITION_FIELDS = ('z_train', 'y_train', 'score_train', 'z_holdout', 'y_holdout', 'holdout_count')


@lru_cache(maxsize=2)
def original_preparation(kind):
    checkpoint, module = source('3582b4ac')
    tree = ast.parse(checkpoint.sources['bayesfilter/inference/quadratic_geometry.py'])
    function_name = '_pilot_q_basis' if kind == 'directions' else 'fit_low_rank_spd_quadratic_geometry'
    original = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name)
    def assigned(node, name):
        return isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name
    first = next(i for i, node in enumerate(original.body) if assigned(node, 'norms' if kind == 'directions' else 'z_finite'))
    last = next(i for i, node in enumerate(original.body) if assigned(node, 'sketch' if kind == 'directions' else 'fit'))
    body = original.body[first:last]
    digest = hashlib.sha256(ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False).encode()).hexdigest()
    parameters = ('directions',) if kind == 'directions' else (
        'z_samples', 'values', 'scores', 'scale_np', 'finite_mask', 'finite_sample_count', 'rng', 'cfg', 'required_finite_samples')
    returned = 'directions' if kind == 'directions' else '{' + ','.join(repr(name) + ':' + name for name in PARTITION_FIELDS) + '}'
    function = ast.FunctionDef(name='original_preparation', args=ast.arguments(posonlyargs=[],
        args=[ast.arg(arg=name) for name in parameters], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=[*body, ast.Return(ast.parse(returned, mode='eval').body)], decorator_list=[])
    namespace = dict(vars(module))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),  # noqa: S102 - unchanged diagnostic AST.
        '3582b4ac:geometry_preparation', 'exec'), namespace)
    return namespace['original_preparation'], digest, checkpoint


def fixture(dimension, case='ordinary'):
    capacity = 24
    raw = np.random.default_rng(908 + dimension).normal(size=(capacity, dimension))
    values = -np.sum(raw * raw, axis=1)
    scores = -2. * raw
    required = dimension + max(0, dimension - 1) + 2
    fraction = .25
    if case == 'partial':
        values[1::5] = np.nan
        scores[2::7, 0] = np.inf
    elif case == 'all_invalid':
        values[:] = np.nan
    elif case in ('threshold', 'below_threshold', 'holdout_cap'):
        keep = required + {'threshold': 0, 'below_threshold': -1, 'holdout_cap': 1}[case]
        values[keep:] = np.nan
    elif case == 'no_holdout':
        fraction = 0.
    finite = np.isfinite(values) & np.all(np.isfinite(scores), axis=1)
    order = np.random.default_rng(405).permutation(int(finite.sum())).astype(np.int32)
    padded = np.pad(order, (0, capacity - len(order)), constant_values=-987)
    args = (tf.constant(raw, D), tf.constant(values, D), tf.constant(scores, D),
            tf.constant(np.linspace(.8, 1.2, dimension), D), tf.constant(padded))
    return args, required, fraction


def partition_reference(args, required, fraction):
    raw, values, scores, scale, order = (value.numpy() for value in args)
    finite = np.isfinite(values) & np.all(np.isfinite(scores), axis=1)
    count = int(finite.sum())
    if count < required:
        return None
    original, _, _ = original_preparation('partition')
    return original(raw, values, scores, scale, finite, count,
        SimpleNamespace(permutation=lambda n: order[:n].copy()), SimpleNamespace(holdout_fraction=fraction), required)


def partition_record(raw):
    training, holdout = int(raw['training_count']), int(raw['holdout_count'])
    return clean({'z_train': raw['z_train'][:training], 'y_train': raw['y_train'][:training],
        'score_train': raw['score_train'][:training], 'z_holdout': raw['z_holdout'][:holdout],
        'y_holdout': raw['y_holdout'][:holdout], 'holdout_count': holdout})


@pytest.mark.parametrize('dimension', [1, 3, 5])
@pytest.mark.parametrize('case', ['ordinary', 'zeros', 'nonfinite', 'empty'])
def test_original_direction_normalization(dimension, case, request):
    raw = np.random.default_rng(708 + dimension).normal(size=(9, dimension))
    if case == 'zeros':
        raw[::2] = 0.
    elif case == 'nonfinite':
        raw[0] = np.inf
        raw[1] = np.nan
        raw[2] = 0.
    elif case == 'empty':
        raw = raw[:0]
    original, digest, checkpoint = original_preparation('directions')
    with np.errstate(invalid='ignore'):
        expected = original(raw)
    records = {}
    for jit in (False, True):
        result = make_direction_preparation_program(dimension, len(raw), jit_compile=jit)(tf.constant(raw, D))
        count = int(result['count'])
        records[str(jit)] = clean(result)
        np.testing.assert_allclose(result['directions'][:count], expected, atol=1e-10, rtol=1e-10, equal_nan=True)
        np.testing.assert_array_equal(result['indices'][:count], np.flatnonzero(np.linalg.norm(raw, axis=1) > 0.))
        assert np.all(result['directions'][count:].numpy() == 0.)
        assert np.all(result['indices'][count:].numpy() == -1)
    save(request, f'geometry-directions-{dimension}-{case}.json', {'input': raw, 'original': expected,
        'records': records, 'original_ast_sha256': digest, 'original_source_sha256': checkpoint.hashes()})


@pytest.mark.parametrize('dimension', [1, 3, 5])
@pytest.mark.parametrize('case', ['ordinary', 'partial', 'all_invalid', 'threshold', 'below_threshold', 'holdout_cap', 'no_holdout'])
def test_original_finite_partition(dimension, case, request):
    args, required, fraction = fixture(dimension, case)
    expected = partition_reference(args, required, fraction)
    records = {}
    mask = np.isfinite(args[1].numpy()) & np.all(np.isfinite(args[2].numpy()), axis=1)
    for jit in (False, True):
        result = make_geometry_partition_program(dimension, 24, required, fraction, jit_compile=jit)(*args)
        records[str(jit)] = clean(result)
        assert int(result['finite_count']) == mask.sum()
        assert bool(result['permutation_valid'])
        assert bool(result['ready']) == (expected is not None)
        np.testing.assert_array_equal(result['finite_indices'][:int(mask.sum())], np.flatnonzero(mask))
        if expected is not None:
            _equal_records(partition_record(result), clean(expected))
            order = np.flatnonzero(mask)[args[-1].numpy()[:int(mask.sum())]]
            split = expected['holdout_count']
            np.testing.assert_array_equal(result['train_indices'][:len(order) - split], order[split:])
            np.testing.assert_array_equal(result['holdout_indices'][:split], order[:split])
        for name in ('z_train', 'y_train', 'score_train', 'z_holdout', 'y_holdout'):
            used = int(result['holdout_count' if 'holdout' in name else 'training_count'])
            assert np.all(result[name][used:].numpy() == 0.)
    _, digest, checkpoint = original_preparation('partition')
    save(request, f'geometry-partition-{dimension}-{case}.json', {'inputs': args,
        'original': expected, 'records': records, 'original_ast_sha256': digest, 'original_source_sha256': checkpoint.hashes()})


@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('case', ['ordinary', 'partial', 'all_invalid'])
def test_design_calls_match_original(batched, case, request):
    args, _, _ = fixture(3)
    center, scale, offsets = tf.constant([.1, .2, .3], D), args[3], args[0]
    with tf.device(center.device):
        calls = tf.Variable(0, dtype=tf.int64)
        positions = tf.Variable(tf.zeros([24, 3], D))
    def callback(point):
        cloud = point if batched else point[None]
        if batched:
            calls.assign_add(1)
            positions.assign(cloud)
        else:
            index = calls.assign_add(1) - 1
            positions.scatter_nd_update(index[None, None], point[None])
        values, scores = -.5 * tf.reduce_sum(cloud * cloud, axis=1), -cloud
        if case == 'partial':
            values = tf.where(cloud[:, 0] > .3, tf.constant(float('nan'), D), values)
        elif case == 'all_invalid':
            scores = tf.fill(tf.shape(cloud), tf.constant(float('nan'), D))
        return (values, scores) if batched else (values[0], scores[0])
    checkpoint, module = source('3582b4ac')
    points = center[None] + offsets * scale[None]
    expected = module._evaluate_values_scores(callback, points.numpy(), batched_value_and_score_fn=callback if batched else None)
    expected_calls, expected_points = int(calls), positions.numpy()
    records = {}
    for jit in (False, True):
        calls.assign(0)
        result = make_geometry_design_program(callback, 3, 24, batched=batched, jit_compile=jit)(center, scale, offsets)
        records[str(jit)] = clean(result)
        _equal_records(clean((result['values'], result['scores'])), clean(expected))
        assert int(calls) == expected_calls == (1 if batched else 24)
        np.testing.assert_allclose(positions, expected_points, atol=1e-10, rtol=1e-10)
    save(request, f'geometry-design-{batched}-{case}.json', {'original': expected, 'records': records,
        'callback_count': expected_calls, 'callback_points': expected_points, 'original_source_sha256': checkpoint.hashes()})


@pytest.mark.parametrize('fault', ['negative', 'duplicate', 'outside'])
def test_invalid_permutation_fails_closed(fault, request):
    args, required, fraction = fixture(3, 'partial')
    bad = args[-1].numpy()
    bad[0] = -1 if fault == 'negative' else (bad[1] if fault == 'duplicate' else 24)
    result = make_geometry_partition_program(3, 24, required, fraction)(*args[:-1], tf.constant(bad))
    save(request, f'geometry-order-{fault}.json', {'inputs': args, 'permutation': bad, 'result': result})
    assert bool(result['sufficient'])
    assert not bool(result['permutation_valid'])
    assert not bool(result['ready'])
    assert int(result['training_count']) == int(result['holdout_count']) == 0


def test_preparation_runtime_inputs_and_enclosing_hlo(request):
    records = {}
    programs = {'directions': make_direction_preparation_program(3, 24),
                'partition': make_geometry_partition_program(3, 24, 7, .25)}
    first, _, _ = fixture(3)
    second, _, _ = fixture(3, 'partial')
    for name, program in programs.items():
        initial = (first[0],) if name == 'directions' else first
        changed = (-first[0],) if name == 'directions' else second
        initial_result, changed_result = program(*initial), program(*changed)
        assert clean(initial_result) != clean(changed_result)
        assert program.experimental_get_tracing_count() == 1
        hlo = program.experimental_get_compiler_ir(*initial)(stage='hlo')
        assert stable_hlo(hlo) == stable_hlo(program.experimental_get_compiler_ir(*changed)(stage='hlo'))
        assert len(re.findall(r'\bparameter\((\d+)\)', hlo[hlo.rfind('\nENTRY '):])) == len(initial)
        outer = tf.function(lambda *args, inner=program: inner(*args), input_signature=program.input_signature,
                            autograph=False, jit_compile=True)
        _equal_records(clean(outer(*changed)), clean(changed_result))
        records[name] = {'initial': initial_result, 'changed': changed_result, 'runtime_inputs': len(initial), 'traces': 1}
    save(request, 'geometry-preparation-runtime-inputs.json', records)


@pytest.mark.parametrize('case', ['ordinary', 'partial', 'holdout_cap', 'all_invalid', 'below_threshold'])
def test_partition_extraction_matches_original_initializer(case, monkeypatch, request):
    args, required, fraction = fixture(3, case)
    checkpoint, module = source('3582b4ac')
    expected = partition_reference(args, required, fraction)
    raw, values, scores, scale, order = (value.numpy() for value in args)
    captured = {}
    class FrozenNumpy:
        random = SimpleNamespace(default_rng=lambda seed: SimpleNamespace(permutation=lambda n: order[:n].copy()))
        def __getattr__(self, name):
            return getattr(np, name)
    def capture(z_train, y_train, score_train, **kwargs):
        frame = inspect.currentframe().f_back
        try:
            captured.update({name: frame.f_locals[name] for name in PARTITION_FIELDS})
        finally:
            del frame
        return {'status': 'fit_nonfinite'}
    def target(point):
        return -.5 * tf.reduce_sum(point ** 2), -point
    with monkeypatch.context() as patch:
        patch.setattr(module, 'np', FrozenNumpy())
        patch.setattr(module, '_pilot_q_basis', lambda *a, **kw: (np.eye(3)[:, :2], {}, ()))
        patch.setattr(module, '_sample_trust_ball', lambda *a, **kw: raw.copy())
        patch.setattr(module, '_evaluate_values_scores', lambda *a, **kw: (values.copy(), scores.copy()))
        patch.setattr(module, '_fit_constrained_quadratic', capture)
        result = module.fit_low_rank_spd_quadratic_geometry(target, np.zeros(3), scale=scale,
            config=module.LowRankSPDQuadraticGeometryConfig(rank=2, sample_count=24,
                min_samples_per_parameter=1, holdout_fraction=fraction))
    if expected is None:
        assert not captured
        assert result.status == 'insufficient_finite_samples'
    else:
        _equal_records(clean(captured), clean({name: expected[name] for name in captured}))
        assert result.status == 'fit_nonfinite'
    save(request, f'geometry-partition-full-original-{case}.json', {'captured': captured,
        'extracted': expected, 'original_source_sha256': checkpoint.hashes()})


def test_extreme_direction_and_holdout_rounding_boundaries(request):
    raw = np.array([[1e-160, 0., 0.], [1e-180, 0., 0.], [1e308, 0., 0.],
                    [1e-154, 1e-154, 0.], [1e-154, 2e-154, 0.], [-0., 0., 0.], [1., -2., 3.]])
    original, _, _ = original_preparation('directions')
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        expected = original(raw)
    records = {}
    for jit in (False, True):
        result = make_direction_preparation_program(3, len(raw), jit_compile=jit)(tf.constant(raw, D))
        records[str(jit)] = clean(result)
    save(request, 'geometry-direction-underflow-records.json', {'raw': raw, 'original': expected, 'records': records})
    for jit in (False, True):
        actual = records[str(jit)]['directions'][:records[str(jit)]['count']]
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10, equal_nan=True)
    args, required, _ = fixture(3)
    for fraction in (np.nextafter(.25, 0.), .25, np.nextafter(.25, 1.)):
        expected_partition = partition_reference(args, required, float(fraction))
        actual = make_geometry_partition_program(3, 24, required, float(fraction))(*args)
        _equal_records(partition_record(actual), clean(expected_partition))
        records[repr(fraction)] = partition_record(actual)
    save(request, 'geometry-preparation-extreme-boundaries.json', {'raw': raw, 'original': expected, 'records': records})


def test_design_and_partition_execute_in_one_xla_program(request):
    args, required, fraction = fixture(3)
    def target(points):
        values = -.5 * tf.reduce_sum(points ** 2, axis=1)
        scores = tf.where(points[:, :1] > .5, tf.constant(float('nan'), D), -points)
        return values, scores
    design = make_geometry_design_program(target, 3, 24, batched=True)
    partition = make_geometry_partition_program(3, 24, required, fraction)
    @tf.function(input_signature=[tf.TensorSpec([3], D), tf.TensorSpec([3], D),
        tf.TensorSpec([24, 3], D), tf.TensorSpec([24], tf.int32)], autograph=False, jit_compile=True)
    def combined(center, scale, offsets, order):
        cloud = design(center, scale, offsets)
        return partition(offsets, cloud['values'], cloud['scores'], scale, order)
    checkpoint, module = source('3582b4ac')
    records = []
    for shift in (0., .2):
        center = tf.fill([3], tf.constant(shift, D))
        points = center[None] + args[0] * args[3][None]
        values, scores = module._evaluate_values_scores(target, points.numpy(), batched_value_and_score_fn=target)
        count = int(np.count_nonzero(np.isfinite(values) & np.all(np.isfinite(scores), axis=1)))
        order = tf.constant(np.pad(np.arange(count)[::-1], (0, 24-count), constant_values=-1), tf.int32)
        expected = partition_reference((args[0], tf.constant(values), tf.constant(scores), args[3], order), required, fraction)
        actual = combined(center, args[3], args[0], order)
        _equal_records(partition_record(actual), clean(expected))
        records.append(clean(actual))
    assert combined.experimental_get_tracing_count() == 1
    hlo = combined.experimental_get_compiler_ir(center, args[3], args[0], order)(stage='hlo')
    assert len(re.findall(r'\bparameter\((\d+)\)', hlo[hlo.rfind('\nENTRY '):])) == 4
    save(request, 'geometry-design-partition-enclosure.json', {'records': records, 'traces': 1,
        'runtime_inputs': 4, 'original_source_sha256': checkpoint.hashes()})


def test_subnormal_products_match_independent_exact_integer_rounding(request):
    rng = random.Random(160922)
    values = [math.ldexp((1 << 52) | rng.getrandbits(52), rng.randint(470, 511) - 1075)
              for _ in range(2048)]
    for units in (0, 1, 2, 3, 2024, 2**20, 2**40, 2**48):
        midpoint = math.ldexp(math.sqrt(units + .5), -537)
        values.extend((math.nextafter(midpoint, 0.), midpoint, math.nextafter(midpoint, math.inf)))
    values.extend(-value for value in values[:20])
    expected = []
    for value in values:
        numerator, denominator = value.as_integer_ratio()
        quotient, remainder = divmod((numerator * numerator) << 1074, denominator * denominator)
        expected.append(quotient + int(2 * remainder > denominator * denominator or (
            2 * remainder == denominator * denominator and quotient % 2)))
    raw = tf.constant(values, D)
    records = {}
    for jit in (False, True):
        program = tf.function(_subnormal_square_units, input_signature=[tf.TensorSpec(raw.shape, D)],
                              autograph=False, jit_compile=jit)
        actual = program(raw).numpy().tolist()
        records[str(jit)] = actual
        assert actual == expected
    with np.errstate(under='ignore'):
        squares = np.asarray(values) ** 2
    numpy_units = []
    for value in squares:
        numerator, denominator = float(value).as_integer_ratio()
        numpy_units.append((numerator << 1074) // denominator)
    assert numpy_units == expected
    save(request, 'geometry-direction-integer-reference.json', {'seed': 160922, 'values': values,
        'exact_reference_units': expected, 'numpy_units': numpy_units, 'records': records,
        'reference': 'Python exact integer ratios, square, divmod and ties-to-even; NumPy product cross-check'})
