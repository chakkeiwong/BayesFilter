"""Incoming preparation versus native preparation with fixed merged score code.

Independent derivative tests separately qualify the merged analytical authority.
These checks isolate the changed preparation at actual public live-input calls.
"""

import hashlib
import importlib
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import sqmc_campaign_tf as current
from bayesfilter.highdim import sqmc_full_lgssm_tf, sqmc_ksc_tf, sqmc_lgssm_tf
from tests.highdim.test_sqmc_campaign_repairs import CONTROLS
from tests.test_filter_repair_sqmc_inputs import BASELINE, ROOT, frozen_inputs


def incoming_module(name, replacements=None):
    """Exact incoming module, with explicitly fixed shared dependencies."""
    source = subprocess.check_output(['git', 'show', f'{BASELINE}:{name}'], cwd=ROOT, text=True)
    alias = 'sqmc_incoming_reference_'+Path(name).stem
    module = types.ModuleType(alias)
    sys.modules[alias] = module
    exec(compile(source, f'{BASELINE}:{name}', 'exec'), module.__dict__)  # noqa: S102 - pinned local diagnostic source
    module.__dict__.update(replacements or {})
    return module, hashlib.sha256(source.encode()).hexdigest()


@pytest.mark.parametrize('family', ['p44', 'diagonal_ar', 'full_matrix', 'ksc_mixture'])
@pytest.mark.parametrize('route', ['iid_dual_cap', 'repaired_permutation'])
def test_live_preparation_at_public_score(family, route, request):
    before_inputs, hashes = frozen_inputs()
    previous, source_hash = incoming_module('bayesfilter/highdim/sqmc_campaign_tf.py',
                                            {'random_inputs': before_inputs})
    module = {'p44': sqmc_lgssm_tf, 'diagonal_ar': sqmc_lgssm_tf,
              'full_matrix': sqmc_full_lgssm_tf, 'ksc_mixture': sqmc_ksc_tf}[family]
    old_spec_module, spec_hash = incoming_module('bayesfilter/highdim/'+Path(module.__file__).name)
    if family == 'ksc_mixture':
        old_spec, spec = old_spec_module.KSCSpec(), module.KSCSpec()
    elif family == 'full_matrix':
        old_spec, spec = old_spec_module.FullLGSSMSpec(family, 3), module.FullLGSSMSpec(family, 3)
    else:
        old_spec, spec = old_spec_module.LGSSMSpec(family, 3), module.LGSSMSpec(family, 3)
    n, horizon = 4*spec.dimension, 2
    dtype = tf.float64
    theta = old_spec.default_theta(dtype)
    observations = tf.reshape(tf.linspace(tf.constant(-.2, dtype), .3, horizon*spec.dimension),
                              [horizon, spec.dimension])
    rows = []
    for seed, shift in ((81102, 0.), (81103, .001)):
        parameters = theta + shift
        expected = previous.value_and_score(old_spec, route, CONTROLS, parameters, observations, seed, n)
        actual = current.value_and_score(spec, route, CONTROLS, parameters, observations, seed, n)
        assert bool(actual[2]) and bool(expected[2])
        for a, b in zip(actual[:2], expected[:2], strict=True):
            np.testing.assert_allclose(a, b, atol=1e-9, rtol=1e-9)
        rows.append({'seed': seed, 'theta': parameters.numpy().tolist(),
                     'original': [x.numpy().tolist() for x in expected],
                     'current': [x.numpy().tolist() for x in actual]})
    settings = current.numerical_settings(CONTROLS)
    owner = current._kernel(spec, route, n, horizon, dtype.name, tuple(sorted(settings.items())), True)
    assert owner.experimental_get_tracing_count() == 1
    inputs = current.random_inputs(route, 81103, n, spec.dimension, horizon, dtype)
    hlo = owner.experimental_get_compiler_ir(theta, *inputs, observations)(stage='hlo')
    assert 'HloModule' in hlo
    graph = owner.get_concrete_function().graph.as_graph_def()
    ops = {node.op for node in graph.node}
    ops.update(node.op for function in graph.library.function for node in function.node_def)
    assert not ops & {'PyFunc', 'PyFuncStateless', 'EagerPyFunc', 'XlaHostCompute'}
    output = Path(request.config.getoption('xmlpath')).parent
    report = {'baseline': BASELINE, 'baseline_input_sources': hashes,
              'baseline_campaign_sha256': source_hash, 'baseline_model_sha256': spec_hash,
              'shared_score': 'merged analytical authority held fixed in both arms',
              'family': family, 'route': route, 'device': actual[0].device,
              'trace_count': 1, 'jit_compile': True, 'rows': rows}
    (output/f'sqmc-endpoint-{family}-{route}.json').write_text(json.dumps(report, indent=2)+'\n')


def test_annealed_actual_enclosing_xla_and_refusals(request):
    from bayesfilter.highdim.ledh_canonical_score_tf import (
        _value_and_analytical_score_impl,
        canonical_value_and_analytical_score,
    )
    from tests.highdim.test_ledh_canonical_score_full import _model

    dtype = tf.float64
    n, d, horizon = 8, 2, 2
    rng = np.random.default_rng(83)
    initial = tf.constant(rng.standard_normal((n, d)), dtype)
    covs = tf.broadcast_to(tf.eye(d, dtype=dtype), [n, d, d])
    noise = tf.constant(rng.standard_normal((horizon, n, d)), dtype)
    observations = tf.constant(rng.standard_normal((horizon, d)), dtype)
    options = {'flow_substeps': 8, 'annealed_stages': 3, 'annealed_seed': 17}

    @tf.function(input_signature=[tf.TensorSpec([1], dtype)], jit_compile=True, autograph=False)
    def compute(theta):
        return canonical_value_and_analytical_score(
            _model(), theta, initial, covs, noise, observations, with_score=True, **options)

    rows = []
    for parameter in (.6, .61):
        theta = tf.constant([parameter], dtype)
        value, score = compute(theta)
        # Independent finite differences of the same realized-index XLA value.
        h = tf.constant([1e-5], dtype)
        expected = (compute(theta+h)[0]-compute(theta-h)[0])/(2e-5)
        np.testing.assert_allclose(score[0], expected, atol=1e-4, rtol=1e-4)
        rows.append({'theta': parameter, 'value': float(value), 'score': float(score[0]),
                     'finite_difference': float(expected)})
    assert compute.experimental_get_tracing_count() == 1
    assert 'HloModule' in compute.experimental_get_compiler_ir(theta)(stage='hlo')
    with pytest.raises(ValueError, match='annealed_stages=1'):
        canonical_value_and_analytical_score(_model(), theta, initial, covs, noise,
            observations, with_score=True, return_trace=True, **options)
    for extra in ({'observation_factor_override': lambda *a: a},
                  {'post_reset_transform': lambda *a: a}):
        with pytest.raises(TypeError, match='unexpected keyword'):
            canonical_value_and_analytical_score(_model(), theta, initial, covs, noise,
                observations, with_score=True, **options, **extra)
        with pytest.raises(ValueError, match='annealed_stages=1'):
            _value_and_analytical_score_impl(_model(), theta, initial, covs, noise,
                observations, with_score=True, reset_policy='contract_e', **options, **extra)
    output = Path(request.config.getoption('xmlpath')).parent
    (output/'sqmc-annealed-xla.json').write_text(json.dumps({
        'rows': rows, 'jit_compile': True, 'trace_count': 1, 'device': value.device,
        'nonclaim': 'finite-program derivative comparison; no LEDH canonical admission'}, indent=2)+'\n')


@pytest.mark.parametrize('horizon,dynamic', [(2, False), (4, True)])
def test_trace_summary_preserves_nomination(horizon, dynamic, monkeypatch, request):
    monkeypatch.syspath_prepend(str(ROOT/'docs/benchmarks'))
    current_runner = importlib.import_module('run_sqmc_expanded_repair')
    previous, source_hash = incoming_module('docs/benchmarks/run_sqmc_expanded_repair.py')
    spec = sqmc_full_lgssm_tf.FullLGSSMSpec('full_matrix', 3)
    theta = spec.default_theta()
    observations = tf.reshape(tf.linspace(tf.constant(-.2, tf.float64), .3, horizon*3), [horizon, 3])
    inputs = current.random_inputs('repaired_permutation', 81102, 12, 3, horizon)
    old = previous.trace_summary_kernel(spec, 'repaired_permutation', CONTROLS, 12, horizon, dynamic_epsilon=dynamic)
    new = current_runner.trace_summary_kernel(spec, 'repaired_permutation', CONTROLS, 12, horizon, dynamic_epsilon=dynamic)
    records = []
    for epsilon in ((32., 64.) if dynamic else (32.,)):
        args = (theta, *inputs, observations) + ((tf.constant(epsilon, tf.float32),) if dynamic else ())
        expected, actual = old(*args), new(*args)
        for a, b in zip(tf.nest.flatten(actual), tf.nest.flatten(expected), strict=True):
            if a.dtype == tf.bool:
                np.testing.assert_array_equal(a, b)
            else:
                np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)
        before, after = previous.summarize_trace(expected), current_runner.summarize_trace(actual)
        assert before['nomination_pass'] == after['nomination_pass']
        assert before['valid'] == after['valid']
        records.append({'epsilon': epsilon, 'original': before, 'current': after})
    assert new.experimental_get_tracing_count() == 1
    compiler_kwargs = {} if dynamic else {'epsilon': None}
    assert 'HloModule' in new.experimental_get_compiler_ir(*args, **compiler_kwargs)(stage='hlo')
    output = Path(request.config.getoption('xmlpath')).parent
    (output/f'sqmc-trace-{horizon}-{dynamic}.json').write_text(json.dumps({
        'incoming_source_sha256': source_hash, 'records': records,
        'device': actual[0].device, 'jit_compile': True, 'trace_count': 1}, indent=2)+'\n')
