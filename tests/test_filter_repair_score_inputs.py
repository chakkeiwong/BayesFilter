"""Pinned-source, stream and complete-consumer diagnostics for input execution."""

import copy
import hashlib
import inspect
import itertools
import json
import os
import subprocess
import types
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.score_study.input_execution_tf import make_score_inputs
from tests.test_filter_repair_gaussian_binding import (
    compare,
    endpoint_case,
    host,
    numerical_record,
    runtime_setup,
    save,
)

ROOT = Path(__file__).resolve().parents[1]
BASELINE = '0025cdf97'
DEPENDENCIES = ('bayesfilter/score_study/adapters.py',
    'bayesfilter/score_study/nonlinear_adapter.py',
    'bayesfilter/score_study/input_execution_tf.py',
    'bayesfilter/ops/stateless_random_tf.py',
    'bayesfilter/score_study/gaussian_tf.py',
    'bayesfilter/score_study/nonlinear_tf.py',
    'bayesfilter/score_study/twist_tf.py')
STREAMS = ('initial', 'process', 'resampling', 'reset_design', 'twist_initial_ancestor')


def hashes():
    return {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in DEPENDENCIES}


def reference(name):
    path = f'bayesfilter/score_study/{name}.py'
    source = subprocess.check_output(['git', 'show', f'{BASELINE}:{path}'], cwd=ROOT, text=True)
    for dependency in DEPENDENCIES[3:]:
        assert subprocess.check_output(['git', 'show', f'{BASELINE}:{dependency}'], cwd=ROOT) == (ROOT/dependency).read_bytes()
    module = types.ModuleType(f'bayesfilter.score_study._input_diagnostic_{name}')
    module.__package__ = 'bayesfilter.score_study'
    exec(compile(source, f'{BASELINE}:{path}', 'exec'), module.__dict__)  # noqa: S102 - pinned diagnostic
    return module, hashlib.sha256(source.encode()).hexdigest()


def eager_inputs(d, N, T, dtype, seeds, twist=False):
    seeds = host(seeds)
    initial = tf.random.stateless_normal([N, d], seeds[0], dtype=dtype)
    process = tf.random.stateless_normal([T, N, d], seeds[1], dtype=dtype)
    uniforms = tf.random.stateless_uniform([T, N], seeds[2], dtype=dtype)
    reset = tf.random.stateless_normal([N, d], seeds[3], dtype=dtype)
    if twist:
        uniforms = tf.concat([tf.random.stateless_uniform([1, N], seeds[4], dtype=dtype), uniforms], 0)
    return initial, process, uniforms, reset


def graph(owner, seeds, request, label):
    concrete = owner.get_concrete_function()
    assert concrete.function_def.attr['_XlaMustCompile'].b
    definition = concrete.graph.as_graph_def()
    nodes = [*definition.node, *(n for f in definition.library.function for n in f.node_def)]
    assert not any('/pfor/' in n.name for n in nodes)
    assert not {n.op for n in nodes} & {'PyFunc', 'PyFuncStateless', 'EagerPyFunc', 'XlaHostCompute'}
    hlo = owner.experimental_get_compiler_ir(seeds)(stage='hlo')
    out = Path(request.config.getoption('xmlpath')).parent
    (out/f'{label}.hlo.txt').write_text(hlo)
    assert owner.experimental_get_tracing_count() == 1
    return {'trace_count': 1, 'jit_compile': True, 'no_host_callback_or_pfor': True,
            'hlo_sha256': hashlib.sha256(hlo.encode()).hexdigest()}


@pytest.mark.parametrize('dtype_name', ['float64', 'float32'])
def test_input_streams(dtype_name, request):
    dtype = tf.as_dtype(dtype_name)
    device = 'CPU' if os.environ['CUDA_VISIBLE_DEVICES'] == '-1' else 'GPU'
    records = []
    for d, N, T in ((1, 7, 3), (3, 8, 2)):
        for twist in (False, True):
            owner = make_score_inputs(d, N, T, dtype_name, twist=twist)
            hlos = []
            previous = None
            for variant in (0, 19):
                seeds = tf.constant([[137+variant, 29], [71, 431+variant], [719+variant, 37],
                                     [91, 523+variant], [13+variant, 863]][:5 if twist else 4], tf.int32)
                expected = eager_inputs(d, N, T, dtype, seeds, twist)
                actual = owner(seeds)
                errors = []
                for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
                    if i == 2:
                        np.testing.assert_array_equal(a.numpy(), b.numpy())
                    else:
                        np.testing.assert_allclose(a.numpy(), b.numpy(), atol=1e-12 if dtype_name == 'float64' else 2e-6, rtol=0)
                    errors.append(float(np.max(np.abs(a.numpy()-b.numpy()))))
                for a, b in zip(actual, owner(seeds), strict=True):
                    np.testing.assert_array_equal(a.numpy(), b.numpy())
                if previous is not None:
                    assert not np.array_equal(actual[0].numpy(), previous)
                previous = actual[0].numpy()
                assert all(device in a.device for a in actual)
                report = graph(owner, seeds, request, f'inputs-{dtype_name}-{d}-{twist}-{variant}')
                hlos.append(report['hlo_sha256'])
                # The conversion must consume the same integer Philox words.
                @tf.function(input_signature=[tf.TensorSpec([2], tf.int32)], jit_compile=True, autograph=False)
                def words(seed, count=T*N*d+3):
                    return tf.random.stateless_uniform([count], seed, minval=None, maxval=None,
                                                       dtype=tf.uint32, alg='philox')
                raw = tf.random.stateless_uniform([T*N*d+3], seeds[0], minval=None, maxval=None,
                                                 dtype=tf.uint32, alg='philox')
                np.testing.assert_array_equal(raw.numpy(), words(seeds[0]).numpy())
                records.append({'d': d, 'N': N, 'T': T, 'twist': twist, 'seeds': host(seeds),
                    'errors': errors, 'raw_words_exact': True, 'uniforms_exact': True, 'graph': report})
            assert hlos[0] == hlos[1]
    with pytest.raises((TypeError, ValueError)):
        make_score_inputs(1, 0, 2)
    with pytest.raises(ValueError, match='float32 or float64'):
        make_score_inputs(1, 8, 2, 'int32')
    owner = make_score_inputs(1, 7, 3, dtype_name, jit_compile=False)
    seeds = tf.constant([[1, 2]]*4, tf.int32)
    assert all(bool(tf.reduce_all(tf.math.is_finite(x))) for x in owner(seeds))
    assert not owner.get_concrete_function().function_def.attr['_XlaMustCompile'].b
    with pytest.raises((TypeError, ValueError)):
        owner(tf.zeros([3, 2], tf.int32))
    save(request, f'score-inputs-{dtype_name}.json', {'baseline': BASELINE, 'device': device,
        'dtype': dtype_name, 'source_sha256': hashes(), 'cases': records,
        'explicit_nonjit_reference_checked': True, 'invalid_configuration_rejected': True})


def seed_table(row, context):
    from bayesfilter.score_study.contracts import seed_pair
    labels = STREAMS if row['proposal'] == 'twist' else STREAMS[:4]
    return tf.constant([seed_pair(master_seed=context['study']['seed'], model=row['model'],
        dataset=row['dataset'], replicate=row['replicate'], stream=name,
        coupling_group=row.get('coupling_group', 'baseline')) for name in labels], tf.int32)


def case(family, proposal, variant=0):
    row, context = endpoint_case(proposal)
    row['twist_power'] = .5
    context['study']['seed'] += variant
    if family.startswith('nonlinear'):
        fixture = json.loads((ROOT/'tests/fixtures/filter_repair_nonlinear_untouched_20260929.json').read_text())['providers'][proposal if proposal in ('ledh', 'sgqf') else 'ledh'][variant]
        settings = context['study']['settings']
        settings.update({key: copy.deepcopy(fixture[key]) for key in ('dimension', 'observation_dimension',
            'particles', 'horizon', 'theta', 'transition_curve', 'observation_curve', 'reference')})
        settings['data_theta'] = list(fixture['theta'])
        context['study']['seed'] = fixture['seed']
        row.update(model='nonlinear_scalar', estimator='nonlinear_analytical')
    else:
        fixture = json.loads((ROOT/'tests/fixtures/filter_repair_score_directions_20260929.json').read_text())
        if family.endswith('directions'):
            context['study']['settings'].update({key: fixture[key] for key in
                ('dimension', 'observation_dimension', 'particles', 'horizon', 'theta')})
            context['study']['settings']['data_theta'] = fixture['theta']
            context['study']['seed'] = fixture['seed']+variant
    row.update(controls=copy.deepcopy(fixture['controls']), sgqf_level=fixture['sgqf_level'])
    if family == 'nonlinear_directions':
        # Frozen mechanics nomination from04806--04821, not a new tuning claim.
        row['controls'].update(reset_sinkhorn_steps=8, reset_balance_steps=8)
    return row, context, fixture


def outcome(endpoint, row, context):
    try:
        return {'result': numerical_record(endpoint(row, context)), 'error': None}
    except (ValueError, tf.errors.InvalidArgumentError) as error:
        return {'result': None, 'error': {'class': type(error).__name__, 'message': str(error)}}


def label_witness(family, proposal, row, context, request):
    """Expose label histories in pinned diagnostic source; require exact witness outputs."""
    from bayesfilter.score_study import gaussian_tf, nonlinear_tf, twist_tf
    s = context['study']['settings']
    d, o, N, T = (s[k] for k in ('dimension', 'observation_dimension', 'particles', 'horizon'))
    dtype = tf.as_dtype(s['dtype'])
    if family == 'nonlinear':
        module, factory = nonlinear_tf, nonlinear_tf.make_particle_filter
        replacements = (
            ('def step(t,x,dx,lw,dlw,value,score,ess,margin):', 'def step(t,x,dx,lw,dlw,value,score,ess,margin,labels):'),
            ('ancestors=tf.searchsorted(cdf,uniforms[t],side="right")', 'ancestors=tf.searchsorted(cdf,uniforms[t],side="right")\n                labels=tf.tensor_scatter_nd_update(labels,[[t]],[ancestors])'),
            ('return t+1,x,dx,lw,dlw,value,score,ess,margin', 'return t+1,x,dx,lw,dlw,value,score,ess,margin,labels'),
            ('tf.cast(N,dtype),q),parallel_iterations=1)', 'tf.cast(N,dtype),q,tf.fill([T,N],-1)),parallel_iterations=1)'),
            ('dtype)),out[7]', 'dtype)),out[7],out[9]'))
        args = (N, T, s['transition_curve'], s['observation_curve'], proposal == 'local_linear', True, s['dtype'], True)
    elif proposal == 'twist':
        module, factory = twist_tf, twist_tf.make_twist_kernel
        replacements = (
            ('derivative,1/tf.reduce_sum(w*w)', 'derivative,1/tf.reduce_sum(w*w),index'),
            ('x,dx,logZ,dlogZ,ess = normalize_and_resample', 'x,dx,logZ,dlogZ,ess,initial_labels = normalize_and_resample'),
            ('def step(t,x,dx,logZ,dlogZ,ess):', 'def step(t,x,dx,logZ,dlogZ,ess,labels):'),
            ('x,dx,increment,derivative,current_ess = normalize_and_resample', 'x,dx,increment,derivative,current_ess,indices = normalize_and_resample'),
            ('tf.minimum(ess,current_ess)', 'tf.minimum(ess,current_ess),tf.tensor_scatter_nd_update(labels,[[t+1]],[indices])'),
            ('(0,x,dx,logZ,dlogZ,ess),parallel_iterations=1)', '(0,x,dx,logZ,dlogZ,ess,tf.tensor_scatter_nd_update(tf.fill([T+1,N],-1),[[0]],[initial_labels])),parallel_iterations=1)'),
            ('return result[3],result[4],result[5]', 'return result[3],result[4],result[5],result[6]'))
        args = (d, o, N, T, row['twist_power'], s['dtype'], True)
    else:
        module, factory = gaussian_tf, gaussian_tf.make_particle_kernel
        replacements = (
            ('def step(t, x, dx, log_weights, dlog_weights, logZ, dlogZ, ess):', 'def step(t, x, dx, log_weights, dlog_weights, logZ, dlogZ, ess, labels):'),
            ('ancestors = tf.searchsorted(cdf, uniforms[0][t], side="right")', 'ancestors = tf.searchsorted(cdf, uniforms[0][t], side="right")\n                labels = tf.tensor_scatter_nd_update(labels, [[t]], [ancestors])'),
            ('return t + 1, x, dx, log_weights, dlog_weights, logZ, dlogZ, ess', 'return t + 1, x, dx, log_weights, dlog_weights, logZ, dlogZ, ess, labels'),
            ('tf.cast(N, dtype)), parallel_iterations=1)', 'tf.cast(N, dtype), tf.fill([T, N], -1)), parallel_iterations=1)'),
            ('return value, score, ess', 'return value, score, ess, result[8]'))
        args = (d, o, N, T, s['dtype'], True, proposal == 'adapted', True)
    original_source = inspect.getsource(getattr(factory, '__wrapped__', factory))
    source = original_source
    for old, new in replacements:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    path = Path(request.config.getoption('xmlpath')).parent/f'{family}-{proposal}-label-diagnostic.py'
    path.write_text(source)
    namespace = dict(module.__dict__)
    exec(compile(source, str(path), 'exec'), namespace)  # noqa: S102 - observable diagnostic only
    diagnostic, ordinary = namespace[factory.__name__](*args), factory(*args)
    seeds = seed_table(row, context)
    inputs = eager_inputs(d, N, T, dtype, seeds, proposal == 'twist')
    changed = make_score_inputs(d, N, T, s['dtype'], twist=proposal == 'twist')(seeds)
    from bayesfilter.score_study.contracts import seed_pair
    data_seed = tf.constant(seed_pair(master_seed=context['study']['seed'], model=row['model'],
        dataset=row['dataset'], replicate=0, stream='observations', coupling_group='common_data'), tf.int32)
    if family == 'nonlinear':
        with tf.device('/CPU:0'):
            obs = nonlinear_tf.make_data_kernel(T, s['transition_curve'], s['observation_curve'], 'float64', True)(
                tf.constant(s['data_theta'], tf.float64), data_seed)
        obs = tf.cast(obs, dtype)
    else:
        obs = gaussian_tf.make_data_kernel(d, o, T, s['dtype'], True)(tf.constant(s['data_theta'], dtype), data_seed)
    theta = tf.constant(s['theta'], dtype)
    records = []
    for values in (inputs, changed):
        operands = (theta, obs, values[0], values[1], values[2])
        actual, witness = ordinary(*operands), diagnostic(*operands)
        records.append({'ordinary': host(actual), 'instrumented': host(witness[:3]), 'labels': host(witness[3])})
    save(request, f'labels-{family}-{proposal}-{context["study"]["seed"]}-{s["dtype"]}.json', {
        'source_sha256': hashlib.sha256(original_source.encode()).hexdigest(),
        'instrumented_sha256': hashlib.sha256(source.encode()).hexdigest(), 'records': records})
    for record in records:
        assert record['ordinary'] == record['instrumented'], 'Instrumentation changed numerical output; not an ordinary-program label witness'
    assert records[0]['labels'] == records[1]['labels'], 'Changed actual ancestor decisions'
    return {'identical_labels': True, 'instrumented_outputs_exact': True,
            'labels': records[0]['labels']}


@pytest.mark.parametrize('family', ['gaussian', 'nonlinear', 'gaussian_directions', 'nonlinear_directions'])
def test_public_inputs(family, request, monkeypatch):
    from bayesfilter.score_study import adapters, input_execution_tf, nonlinear_adapter
    nonlinear = family.startswith('nonlinear')
    module = nonlinear_adapter if nonlinear else adapters
    old, source_hash = reference('nonlinear_adapter' if nonlinear else 'adapters')
    endpoint_name = 'evaluate_nonlinear' if nonlinear else 'evaluate_gaussian'
    current, previous = getattr(module, endpoint_name), getattr(old, endpoint_name)
    for target in (module, old):
        monkeypatch.setattr(target, 'configure_runtime', runtime_setup)
    proposals = ('ledh', 'sgqf') if family.endswith('directions') else (
        ('ekf', 'ukf', 'bootstrap', 'prior_sis', 'local_linear') if nonlinear else
        ('kalman', 'ukf', 'bootstrap', 'prior_sis', 'adapted', 'adapted_sis', 'twist'))
    factory = input_execution_tf.make_score_inputs
    seen = []
    def observed(*args, **kwargs):
        owner = factory(*args, **kwargs)
        seen.append(owner)
        return owner
    monkeypatch.setattr(input_execution_tf, 'make_score_inputs', observed)
    records = []
    dtypes = ('float64',) if family.endswith('directions') else ('float64', 'float32')
    for proposal, variant, dtype_name in itertools.product(proposals, (0, 1), dtypes):
        row, context, fixture = case(family, proposal, variant)
        context['study']['settings']['dtype'] = dtype_name
        expected = outcome(previous, row, context)
        actual = outcome(current, row, context)
        replay = outcome(current, row, context)
        compare(actual, expected, 1e-9 if dtype_name == 'float64' else 5e-5)
        assert actual == replay
        if not family.endswith('directions'):
            assert actual['error'] is None
        frozen = eager_inputs(context['study']['settings']['dimension'],
            context['study']['settings']['particles'], context['study']['settings']['horizon'],
            tf.as_dtype(dtype_name), seed_table(row, context), proposal == 'twist')
        with monkeypatch.context() as patch:
            patch.setattr(input_execution_tf, 'make_score_inputs', lambda *_, values=frozen, **__: lambda _: values)
            controlled = outcome(current, row, context)
            assert controlled == expected
        records.append({'proposal': proposal, 'variant': variant, 'dtype': dtype_name, 'live_original': expected,
            'live_candidate': actual, 'frozen_inputs_exact': True})
        if family in ('gaussian', 'nonlinear') and proposal in ('bootstrap', 'adapted', 'local_linear', 'twist'):
            records[-1]['discrete_witness'] = label_witness(family, proposal, row, context, request)
        if family.endswith('directions'):
            # Preserve the previously qualified healthy frozen fixture too.
            from bayesfilter.score_study import gaussian_tf, nonlinear_tf
            numerical = nonlinear_tf if nonlinear else gaussian_tf
            dtype = tf.float64
            arrays = tuple(tf.constant(fixture[key], dtype) for key in
                           ('initial', 'process', 'resampling_uniforms', 'reset_design'))
            seeds = host(seed_table(row, context))
            lookup = {tuple(seed): tensor for seed, tensor in zip(seeds, arrays, strict=True)}
            def frozen_random(shape, seed, dtype, lookup=lookup):
                result = lookup[tuple(seed)]
                assert result.shape == shape and result.dtype == dtype
                return result
            with monkeypatch.context() as patch:
                patch.setattr(numerical, 'make_data_kernel', lambda *_, values=fixture['observations'], dtype=dtype, **__: lambda *_: tf.constant(values, dtype))
                patch.setattr(tf.random, 'stateless_normal', frozen_random)
                patch.setattr(tf.random, 'stateless_uniform', frozen_random)
                patch.setattr(input_execution_tf, 'make_score_inputs', lambda *_, values=arrays, **__: lambda _: values)
                old_frozen, new_frozen = outcome(previous, row, context), outcome(current, row, context)
                assert old_frozen == new_frozen and new_frozen['error'] is None
            records[-1]['healthy_frozen_record'] = new_frozen
    assert seen and all(owner.experimental_get_tracing_count() == 1 for owner in seen)
    # Invalid numerical values must still reach the same public refusal.
    row, context, _ = case(family, proposals[0])
    context['study']['settings']['theta'][0] = float('nan')
    expected, actual = outcome(previous, row, context), outcome(current, row, context)
    assert actual == expected and actual['error'] is not None
    save(request, f'score-inputs-public-{family}.json', {'baseline': BASELINE,
        'baseline_adapter_sha256': source_hash, 'source_sha256': hashes(), 'cases': records,
        'single_trace': True, 'invalidity_preserved': True, 'deferred_families_not_executed': True})
