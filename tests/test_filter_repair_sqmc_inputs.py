"""Independent incoming-parent stream and preparation reference comparisons."""

import ast
import hashlib
import json
import subprocess
import types
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import sqmc_campaign_tf as current
from bayesfilter.highdim.sqmc_full_lgssm_tf import FullLGSSMSpec
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint

ROOT = Path(__file__).resolve().parents[1]
BASELINE = '023e10610'


def frozen_inputs():
    """Load exact input-only definitions; the diagnostic calls no score code."""
    path = 'bayesfilter/highdim/sqmc_campaign_tf.py'
    source = subprocess.check_output(['git', 'show', f'{BASELINE}:{path}'], cwd=ROOT, text=True)
    frozen = FrozenCheckpoint(BASELINE, 'sqmc_input')
    sqmc = frozen.load('bayesfilter.highdim.sqmc_tf')
    nodes = [node for node in ast.parse(source).body if
             isinstance(node, ast.FunctionDef) and node.name in ('route_settings', 'random_inputs') or
             isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ROUTES' for t in node.targets)]
    namespace = {'tf': tf, 'randomized_halton_gaussian': sqmc.randomized_halton_gaussian,
                 'randomized_halton_joint': sqmc.randomized_halton_joint}
    exec(compile(ast.Module(nodes, type_ignores=[]), f'{BASELINE}:{path}', 'exec'), namespace)  # noqa: S102 - exact pinned diagnostic definitions
    return namespace['random_inputs'], {**frozen.hashes(), path: hashlib.sha256(source.encode()).hexdigest()}


def save(request, name, report):
    output = Path(request.config.getoption('xmlpath')).parent
    (output/name).write_text(json.dumps(report, indent=2)+'\n')


@pytest.mark.parametrize('dtype_name', ['float32', 'float64'])
@pytest.mark.parametrize('mode', ['ordinary', 'graph', 'xla'])
@pytest.mark.parametrize('route', ['iid_dual_cap', 'repaired_permutation'])
def test_original_live_clouds(route, mode, dtype_name, request):
    before, sources = frozen_inputs()
    dtype = tf.as_dtype(dtype_name)
    rows = []
    for n, d, horizon in ((12, 3, 2), (8, 2, 3)):
        # The incoming Halton helper int-coerces Python seeds. Rebuild only the
        # frozen comparison for each seed; the new owner retains one signature.
        program = current._input_kernel(route == 'iid_dual_cap', n, d, horizon, dtype_name,
                                        mode == 'xla', mode != 'graph')
        for seed in (70003, -17):
            args = (route, seed, n, d, horizon, dtype)
            if mode == 'ordinary':
                expected = before(*args)
                actual = current.random_inputs(*args)
            else:
                old = tf.function(lambda args=args: before(*args), input_signature=[],
                                  jit_compile=mode == 'xla', autograph=False)
                new = tf.function(lambda args=args: current.random_inputs(*args, jit_compile=mode == 'xla'),
                                  input_signature=[], jit_compile=mode == 'xla', autograph=False)
                expected, actual = old(), new()
            errors = []
            for a, b in zip(actual, expected, strict=True):
                np.testing.assert_allclose(a, b, atol=3e-6 if dtype == tf.float32 else 1e-12,
                                           rtol=3e-6 if dtype == tf.float32 else 1e-12)
                errors.append(float(tf.reduce_max(tf.abs(a-b))))
            # Sorting rows and ancestor selection inputs must not change.
            np.testing.assert_array_equal(tf.argsort(actual[2], stable=True), tf.argsort(expected[2], stable=True))
            np.testing.assert_array_equal(actual[2], expected[2])
            replay = current.random_inputs(*args) if mode == 'ordinary' else new()
            for a, b in zip(actual, replay, strict=True):
                np.testing.assert_array_equal(a, b)
            rows.append({'seed': seed, 'n': n, 'd': d, 'horizon': horizon, 'errors': errors,
                         'original': [x.numpy().tolist() for x in expected],
                         'current': [x.numpy().tolist() for x in actual]})
        assert program.experimental_get_tracing_count() == 1
        graph = program.get_concrete_function().graph.as_graph_def()
        assert any('While' in node.op for node in graph.node)
        assert not any('PyFunc' in node.op for node in graph.node)
        if mode != 'graph':
            assert 'HloModule' in program.experimental_get_compiler_ir(tf.constant(70003, tf.int64))(stage='hlo')
    save(request, f'sqmc-inputs-{route}-{mode}-{dtype_name}.json',
         {'baseline': BASELINE, 'baseline_sources': sources, 'device': actual[0].device,
          'route': route, 'mode': mode, 'dtype': dtype_name, 'rows': rows})


@pytest.mark.parametrize('dimension', [2, 3, 10])
@pytest.mark.parametrize('dtype_name', ['float32', 'float64'])
def test_full_matrix_preparation_preserves_coordinates(dimension, dtype_name):
    frozen = FrozenCheckpoint(BASELINE, 'sqmc_full')
    namespace = types.ModuleType(frozen.prefix+'.bayesfilter.highdim')
    namespace.__path__ = []
    frozen.modules['bayesfilter.highdim'] = namespace
    before = frozen.load('bayesfilter.highdim.sqmc_full_lgssm_tf').FullLGSSMSpec('full_matrix', dimension)
    after = FullLGSSMSpec('full_matrix', dimension)
    dtype = tf.as_dtype(dtype_name)
    expected, actual = before.default_theta(dtype), after.default_theta(dtype)
    if dtype == tf.float32:
        np.testing.assert_array_max_ulp(actual.numpy(), expected.numpy(), maxulp=1)
    else:
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-14)
    assert before.parameter_names == after.parameter_names
    signature = [tf.TensorSpec([after.parameter_count], dtype)]*2
    old = tf.function(before.parts, input_signature=signature, jit_compile=True, autograph=False)
    new = tf.function(after.parts, input_signature=signature, jit_compile=True, autograph=False)
    for shift in (0., .01):
        theta = expected + tf.cast(shift, dtype)
        tangent = tf.sin(tf.cast(tf.range(after.parameter_count), dtype))
        a, b = new(theta, tangent), old(theta, tangent)
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_allclose(a[key], b[key], atol=1e-7 if dtype == tf.float32 else 1e-14,
                                      rtol=1e-7 if dtype == tf.float32 else 1e-14, err_msg=key)
    assert new.experimental_get_tracing_count() == 1


def test_input_invalid_contract():
    with pytest.raises(ValueError, match='unknown SQMC route'):
        current.random_inputs('missing', 1, 12, 3, 2)
    for sizes in ((0, 3, 2), (12, 0, 2), (12, 3, 0)):
        with pytest.raises(ValueError, match='must be positive'):
            current.random_inputs('iid_dual_cap', 1, *sizes)
    with pytest.raises(TypeError, match='integer dtype'):
        current.random_inputs('iid_dual_cap', 1.2, 12, 3, 2)


def test_original_halton_seed_coercions():
    before, _ = frozen_inputs()
    for seed in (2**80+91, '73', 3.8, tf.constant(-17, tf.int32)):
        args = ('repaired_permutation', seed, 8, 2, 2, tf.float64)
        expected, actual = before(*args), current.random_inputs(*args)
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)
        np.testing.assert_array_equal(actual[2], expected[2])


@pytest.mark.parametrize('dimension', [1, 3, 10, 17])
@pytest.mark.parametrize('dtype_name', ['float32', 'float64'])
def test_diagonal_default_preparation(dimension, dtype_name):
    from bayesfilter.highdim.sqmc_lgssm_tf import LGSSMSpec
    frozen = FrozenCheckpoint(BASELINE, 'sqmc_diagonal_default')
    before = frozen.load('bayesfilter.highdim.sqmc_lgssm_tf').LGSSMSpec('diagonal_ar', dimension)
    after = LGSSMSpec('diagonal_ar', dimension)
    dtype = tf.as_dtype(dtype_name)
    expected, actual = before.default_theta(dtype), after.default_theta(dtype)
    if dtype == tf.float32:
        np.testing.assert_array_max_ulp(actual.numpy(), expected.numpy(), maxulp=1)
    else:
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize('dtype_name', ['float32', 'float64'])
def test_halton_digit_boundaries_and_campaign_sizes(dtype_name, request):
    from bayesfilter.highdim.sqmc_tf import randomized_halton_joint
    frozen = FrozenCheckpoint(BASELINE, 'halton_boundaries')
    before = frozen.load('bayesfilter.highdim.sqmc_tf').randomized_halton_joint
    dtype = tf.as_dtype(dtype_name)
    records = []
    for n, d in ((3, 2), (9, 3), (27, 3), (81, 3), (1024, 3), (1008, 3), (1020, 10)):
        options = {'num_particles': n, 'state_dimension': d, 'seed': 81102, 'salt': 3001, 'dtype': dtype}
        # Ordinary GPU pow rounds 19**2 upward, corrupting the integer digit
        # at indices 361/722. The isolated attribution check preserves that
        # failure and verifies CPU/XLA digits against exact integer division.
        with tf.device('/CPU:0'):
            eager = before(**options)
        original = tf.function(lambda opts=options: before(**opts), input_signature=[], jit_compile=True, autograph=False)
        native = tf.function(lambda opts=options: randomized_halton_joint(**opts), input_signature=[], jit_compile=True, autograph=False)
        preserved = tf.function(lambda opts=options: randomized_halton_joint(**opts, ordinary_stream=True),
                                input_signature=[], jit_compile=True, autograph=False)
        comparisons = []
        for context, expected, actual in (('ordinary', eager, preserved()), ('xla', original(), native())):
            errors = [float(tf.reduce_max(tf.abs(a-b))) for a, b in zip(actual, expected, strict=True)]
            mismatch = np.argwhere(np.abs(actual[0].numpy()-expected[0].numpy()) >
                                  (2e-7 if dtype == tf.float32 else 1e-14))
            comparisons.append({'context': context, 'maximum_absolute_errors': errors,
                                'joint_mismatches': [{'index': index.tolist(),
                                    'actual': float(actual[0][tuple(index)]),
                                    'expected': float(expected[0][tuple(index)])}
                                    for index in mismatch[:20]]})
            save(request, f'halton-boundaries-{dtype_name}-progress.json', {
                'baseline': BASELINE, 'records': records,
                'current': {'n': n, 'd': d, 'comparisons': comparisons},
                'dtype': dtype_name, 'device': actual[0].device})
            for a, b in zip(actual, expected, strict=True):
                np.testing.assert_allclose(a, b, atol=2e-7 if dtype == tf.float32 else 1e-14,
                                           rtol=2e-7 if dtype == tf.float32 else 1e-14)
            np.testing.assert_array_equal(tf.argsort(actual[0][:, 0], stable=True),
                                          tf.argsort(expected[0][:, 0], stable=True))
            np.testing.assert_array_equal(actual[1], expected[1])
        records.append({'n': n, 'd': d, 'comparisons': comparisons})
    save(request, f'halton-boundaries-{dtype_name}.json', {
        'baseline': BASELINE, 'baseline_sources': frozen.hashes(), 'records': records,
        'dtype': dtype_name, 'device': actual[0].device, 'seed': 81102, 'salt': 3001,
        'ordinary_reference_device': eager[0].device,
        'ordinary_gpu_reference_defect': 'pow(19,2) rounds upward; run05200 attribution'})


def test_halton_gpu_radix_digit_attribution(request):
    from tensorflow_probability.python.mcmc import (
        sample_halton_sequence_lib as reference,
    )

    from bayesfilter.highdim.sqmc_tf import randomized_halton_joint

    frozen = FrozenCheckpoint(BASELINE, 'halton_radix_attribution')
    before = frozen.load('bayesfilter.highdim.sqmc_tf').randomized_halton_joint
    options = {'num_particles': 1020, 'state_dimension': 10, 'seed': 81102,
               'salt': 3001, 'dtype': tf.float64}

    def digits():
        radixes = reference._PRIMES[:11, None]
        sizes = reference._base_expansion_size(1020, radixes, tf.float64)
        exponents = tf.tile([tf.range(np.max(sizes), dtype=tf.float64)], [11, 1])
        mask = exponents < sizes
        weights = radixes ** tf.where(mask, exponents, 0.)
        indices = reference._get_indices(1020, None, tf.float64)
        coefficients = tf.math.floordiv(indices, weights) * tf.cast(mask, tf.float64)
        coefficients %= radixes
        integer_weights = tf.cast(tf.round(weights), tf.int64)
        exact = (tf.math.floordiv(tf.cast(indices, tf.int64), integer_weights) *
                 tf.cast(mask, tf.int64)) % tf.constant(radixes, tf.int64)
        return weights, tf.cast(coefficients, tf.int64), exact

    with tf.device('/CPU:0'):
        cpu = before(**options)
        cpu_digits = digits()
    with tf.device('/GPU:0'):
        gpu = before(**options)
        gpu_digits = digits()
        xla_digits = tf.function(digits, input_signature=[], jit_compile=True, autograph=False)()
        ordinary = tf.function(lambda: randomized_halton_joint(**options, ordinary_stream=True),
                               input_signature=[], jit_compile=True, autograph=False)()
        original_xla = tf.function(lambda: before(**options), input_signature=[],
                                  jit_compile=True, autograph=False)()
        current_xla = tf.function(lambda: randomized_halton_joint(**options), input_signature=[],
                                 jit_compile=True, autograph=False)()
    np.testing.assert_array_equal(cpu_digits[1], cpu_digits[2])
    np.testing.assert_array_equal(xla_digits[1], xla_digits[2])
    mismatches = np.argwhere(gpu_digits[1].numpy() != gpu_digits[2].numpy())
    np.testing.assert_array_equal(mismatches, [[360, 7, 2], [721, 7, 2]])
    for actual, expected in zip(ordinary, cpu, strict=True):
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-14)
    for actual, expected in zip(current_xla, original_xla, strict=True):
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-14)
    np.testing.assert_array_equal(ordinary[1], cpu[1])
    np.testing.assert_array_equal(current_xla[1], original_xla[1])
    save(request, 'halton-gpu-radix-attribution.json', {
        'baseline': BASELINE, 'baseline_sources': frozen.hashes(),
        'gpu': gpu[0].device, 'cpu': cpu[0].device,
        'radix19_squared': {name: float(value[0][7, 2]) for name, value in
                            [('cpu', cpu_digits), ('gpu', gpu_digits), ('xla', xla_digits)]},
        'incorrect_gpu_digit_indices': mismatches.tolist(),
        'maximum_original_gpu_cpu_difference': float(tf.reduce_max(tf.abs(gpu[0]-cpu[0]))),
        'maximum_repair_cpu_difference': float(tf.reduce_max(tf.abs(ordinary[0]-cpu[0]))),
        'maximum_enclosed_difference': float(tf.reduce_max(tf.abs(current_xla[0]-original_xla[0]))),
        'cpu_and_xla_digits_equal_exact_integer_reference': True})
