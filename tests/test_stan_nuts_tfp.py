"""Native TFP API and CPU/XLA parity diagnostics; not posterior evidence."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['BAYESFILTER_PRELOAD_CUSTOM_OP'] = '0'
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
import ast
from pathlib import Path
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from docs.benchmarks import stan_nuts_tfp as native
from docs.benchmarks import stan_nuts_warmup as adaptation


def target(q):
    x = q[:, 0]
    y = q[:, 1] - .2*(x*x-1.)
    return -.5*(x*x+2.*y*y)


def equal(a, b):
    tf.debugging.assert_near(a, b, rtol=2e-12, atol=2e-12)


def inputs(metric):
    q = tf.constant([[.2, -.1], [-.4, .7]], tf.float64)
    e = tf.constant([.3, .15], tf.float64)
    if metric == 'dense':
        a = tf.broadcast_to(tf.constant([[1.2, 0.], [.3, .8]], tf.float64), [2, 2, 2])
        m = tf.matmul(a, a, transpose_b=True)
    else:
        a = tf.linalg.diag(tf.ones([2, 2], tf.float64) if metric == 'unit' else
                           tf.constant([[1.2, .8], [.7, 1.3]], tf.float64))
        m = tf.square(tf.linalg.diag_part(a))
    return q, e, a, m, tf.constant([37, 51], tf.int32)


@pytest.mark.parametrize('metric', ['unit', 'diag', 'dense'])
def test_native_equivalence_and_coordinate_cache(metric):
    q, e, a, m, seed = inputs(metric)
    k = native.StanNoUTurnSampler(target, 2, chains=2, step_size=e, inverse_metric=m,
                                  metric=metric, max_tree_depth=4, seed_state=seed)
    def transformed(z):
        # Independent affine reference including the constant log Jacobian.
        value = target(tf.linalg.matvec(a, z))
        return value+tf.reduce_sum(tf.math.log(tf.linalg.diag_part(a)), -1)
    stock = tfp.mcmc.NoUTurnSampler(transformed, e[:, None], max_tree_depth=4, parallel_iterations=1)

    @tf.function(input_signature=[tf.TensorSpec([2, 2], tf.float64)],
                 autograph=False, jit_compile=True)
    def compare(q):
        kr = k.bootstrap_results(q)
        z = tf.linalg.triangular_solve(a, q[..., None])[..., 0]
        sr = stock.bootstrap_results(z)
        rng = seed
        # Small fixed test loop; native production chains use sample_chain.
        for _ in range(3):
            keys = tf.random.experimental.stateless_split(rng, 2); rng = keys[0]
            q, kr = k.one_step(q, kr)
            z, sr = stock.one_step(z, sr, seed=keys[1])
        with tf.GradientTape() as tape:
            tape.watch(z);value = transformed(z)
        return q, kr, tf.linalg.matvec(a, z), sr, value, tape.gradient(value, z), rng

    actual, result, expected, sr, value, grad, rng = compare(q)
    nr = k.native_results(result)
    equal(actual, expected);equal(nr.target_log_prob, value)
    equal(nr.grads_target_log_prob[0], grad);equal(nr.log_accept_ratio, sr.log_accept_ratio)
    tf.debugging.assert_equal(nr.leapfrogs_taken, sr.leapfrogs_taken)
    tf.debugging.assert_equal(result.seed, rng);native.require_valid(result)
    assert isinstance(k, tfp.mcmc.TransitionKernel)
    assert 'HloModule' in compare.experimental_get_compiler_ir(q)(stage='optimized_hlo')
    assert compare.experimental_get_tracing_count() == 1


def runner(k, count, resume=False, explicit=False):
    qs = tf.TensorSpec([2, 2], tf.float64)
    sig = [qs]
    if resume:
        sig.append(tf.nest.map_structure(lambda x: tf.TensorSpec(x.shape, x.dtype),
                                        k.bootstrap_results(tf.zeros([2, 2], tf.float64))))
    if explicit: sig.append(tf.TensorSpec([2], tf.int32))
    @tf.function(input_signature=sig, autograph=False, jit_compile=True)
    def run(q, *extra):
        return tfp.mcmc.sample_chain(count, q, kernel=k,
            previous_kernel_results=extra[0] if resume else None,
            seed=extra[-1] if explicit else None,
            trace_fn=lambda _, kr: kr.valid, return_final_kernel_results=True, parallel_iterations=1)
    return run


def test_sample_chain_seed_reproduction_and_exact_continuation():
    q, e, a, m, seed = inputs('diag')
    k = native.StanNoUTurnSampler(target, 2, chains=2, step_size=e, inverse_metric=m,
                                  max_tree_depth=4, seed_state=seed)
    first = runner(k, 5)(q)
    second = runner(k, 5, resume=True)(first.all_states[-1], first.final_kernel_results)
    full_fn = runner(k, 10);full = full_fn(q)
    tf.debugging.assert_equal(tf.concat([first.all_states, second.all_states], 0), full.all_states)
    tf.debugging.assert_equal(second.final_kernel_results.seed, full.final_kernel_results.seed)
    native.require_valid(second.final_kernel_results)
    run = runner(k, 4, explicit=True)
    x = run(q, seed);y = run(q, seed);z = run(q, seed+tf.constant([1, 0]))
    tf.debugging.assert_equal(x.all_states, y.all_states)
    assert bool(tf.reduce_any(x.all_states != z.all_states))
    assert bool(tf.reduce_any(x.all_states[:, 0] != x.all_states[:, 1]))
    assert 'HloModule' in run.experimental_get_compiler_ir(q, seed)(stage='optimized_hlo')


@pytest.mark.parametrize('metric', ['diag', 'dense'])
def test_stan_style_warmup_and_compiled_factory(metric):
    q, e, a, m, seed = inputs(metric)
    cfg = adaptation.Config(2, chains=2, warmup=20, draws=4, depth=3, metric=metric,
                            initial_buffer=5, terminal_buffer=5, window=5)
    warm, sample = native.make_functions(tf, tfp, target, cfg)
    warmed = native.require_valid(warm(q, seed))
    k = native.StanNoUTurnSampler.from_warmup(target, warmed, cfg)
    result = runner(k, 4)(warmed.position)
    compiled = native.require_valid(sample(*warmed[:4]))
    equal(result.all_states, compiled.draws)
    tf.debugging.assert_equal(result.final_kernel_results.seed, compiled.seed)
    assert sample.experimental_get_tracing_count() == 1
    assert 'HloModule' in sample.experimental_get_compiler_ir(*warmed[:4])(stage='optimized_hlo')
    next_chunk = sample(compiled.position, warmed.step_size, warmed.metric_factor, compiled.seed)
    _, long = native.make_functions(tf, tfp, target, adaptation.Config(**{**cfg.__dict__, 'draws':8}))
    equal(tf.concat((compiled.draws, next_chunk.draws), 0), long(*warmed[:4]).draws)
    with pytest.raises(RuntimeError):
        native.StanNoUTurnSampler.from_warmup(target, warmed._replace(valid=tf.zeros([2], tf.bool)), cfg)


@pytest.mark.parametrize('bad', ['epsilon', 'metric', 'state', 'no_seed'])
def test_invalid_results_fail_closed(bad):
    q, e, a, m, seed = inputs('diag')
    if bad == 'epsilon': e = -e
    if bad == 'metric': m = -m
    if bad == 'state': q = q*float('nan')
    k = native.StanNoUTurnSampler(target, 2, chains=2, step_size=e, inverse_metric=m,
          max_tree_depth=2, seed_state=None if bad == 'no_seed' else seed)
    result = runner(k, 2)(q)
    with pytest.raises(RuntimeError): native.require_valid(result.final_kernel_results)


def test_rejected_nonfinite_proposals_keep_valid_state():
    def bounded(q):
        return tf.where(tf.reduce_all(tf.abs(q)<1., -1), -.5*tf.reduce_sum(q*q, -1),
                        tf.constant(float('nan'), tf.float64))
    k = native.StanNoUTurnSampler(bounded, 2, chains=2, step_size=1e6,
                                  metric='unit', max_tree_depth=2, seed_state=[17, 31])
    result = runner(k, 4)(tf.zeros([2, 2], tf.float64))
    native.require_valid(result.final_kernel_results)
    tf.debugging.assert_equal(result.all_states, tf.zeros_like(result.all_states))
    assert bool(tf.reduce_all(k.native_results(result.final_kernel_results).has_divergence))


def test_migration_copy_and_no_direct_tree_dependency():
    k = native.StanNoUTurnSampler(target, 2, chains=2, step_size=.2)
    equal(k.copy(step_size=.4).step_size, tf.constant([.4,.4], tf.float64))
    for kw in [{'step_size_jitter':.1}, {'seed_state':[[17,0],[31,0]]},
               {'max_tree_depth':21}, {'metric':'bad'}, {'dimension':0}]:
        with pytest.raises(ValueError):
            native.StanNoUTurnSampler(target, **{'dimension':2, 'chains':2, **kw})
    with pytest.raises(ValueError): k.one_step(tf.zeros([2,2],tf.float64), None, seed=123)
    source=Path(native.__file__).read_text();tree=ast.parse(source)
    assert 'stan_nuts.Kernel' not in source and 'port.Kernel' not in source
    assert not any(isinstance(n,ast.Import) and any(a.name=='numpy' for a in n.names) for n in ast.walk(tree))
