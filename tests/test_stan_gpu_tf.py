"""Diagnostic-only CPU/XLA checks for generic TF target integration."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['BAYESFILTER_PRELOAD_CUSTOM_OP'] = '0'
from dataclasses import replace
import pytest
from docs.benchmarks import stan_nuts as core
from docs.benchmarks.stan_gpu_tf import api
from docs.benchmarks import benchmark_nuts_xla as base


@pytest.fixture(scope='module')
def tf():
    args = base.parser().parse_args(['--device', 'cpu', '--output', '/tmp/stan-gpu-tf-test'])
    return base.setup(args)[0]


@pytest.mark.parametrize('chains', [1, 4, 16])
@pytest.mark.parametrize('normal', [False, True])
def test_random_stream_identity(tf, chains, normal):
    cfg = core.Config(3, chains=chains)
    @tf.function(input_signature=[tf.TensorSpec([chains, 2], tf.int32),
        tf.TensorSpec([chains], tf.bool)], jit_compile=True, autograph=False)
    def compare(seed, mask):
        target = lambda q: -.5*tf.reduce_sum(q*q, -1)
        a = core.Kernel(tf, target, cfg).random(seed, mask, normal=normal, size=(3,))
        b = api.StaticChainKernel(tf, target, cfg).random(seed, mask, normal=normal, size=(3,))
        return a, b
    seed = tf.constant([[31+i, 7*i] for i in range(chains)], tf.int32)
    mask = tf.constant([i % 2 == 0 for i in range(chains)])
    a, b = compare(seed, mask)
    for x, y in zip(a, b):
        assert bool(tf.reduce_all(x == y))


def model(tf):
    return lambda q, x, y: -.5*tf.reduce_sum(q*q, -1) - .5*tf.reduce_sum(
        (tf.matmul(q, x, transpose_b=True)-y)**2, -1)/tf.cast(x.shape[0], tf.float64)


@pytest.mark.parametrize('metric', ['unit', 'diag', 'dense'])
def test_full_fit_dynamic_model_and_baseline(tf, metric):
    cfg = core.Config(2, chains=2, warmup=30, draws=12, depth=4, metric=metric,
                      initial_buffer=2, terminal_buffer=1, window=1)
    specs = [tf.TensorSpec([4, 2], tf.float64), tf.TensorSpec([4], tf.float64)]
    candidate = api.make_functions(tf, model(tf), cfg, specs)
    baseline = api.make_functions(tf, model(tf), cfg, specs, rng_layout='map')
    q = tf.constant([[.2, -.4], [.8, .3]], tf.float64)
    m = tf.eye(2, batch_shape=[2], dtype=tf.float64) if metric == 'dense' else tf.ones_like(q)
    data = (tf.constant([[1., 0.], [0., 1.], [.5, .3], [-.2, 1.]], tf.float64),
            tf.constant([.1, .2, -.1, .4], tf.float64))
    args = (q, tf.ones_like(q, tf.bool), tf.ones([2], tf.float64), m,
            tf.constant([[51, 0], [73, 0]], tf.int32), *data)
    a, b = candidate.fit(*args), baseline.fit(*args)
    assert bool(tf.reduce_all(a.valid)) and bool(tf.reduce_all(b.valid))
    for x, y in zip(tf.nest.flatten(a), tf.nest.flatten(b)):
        if x.dtype.is_floating:
            assert bool(tf.reduce_all(tf.abs(x-y) <= 2e-12*(1.+tf.abs(y))))
        else:
            assert bool(tf.reduce_all(x == y))
    assert bool(tf.reduce_all(a.sampling.seed == b.sampling.seed))
    v, g = candidate.value_gradient(q, *data)
    expected = -q-tf.matmul(tf.matmul(q, data[0], transpose_b=True)-data[1], data[0])/4.
    assert float(tf.reduce_max(tf.abs(g-expected))) < 2e-12
    changed = candidate.fit(*args[:-1], data[1]+1.)
    assert bool(tf.reduce_all(changed.valid))
    assert not bool(tf.reduce_all(changed.sampling.draws == a.sampling.draws))
    assert candidate.fit.experimental_get_tracing_count() == 1
    assert not base.verify_graph(candidate.fit)['forbidden_ops']
    hlo = candidate.fit.experimental_get_compiler_ir(*args)(stage='optimized_hlo')
    assert 'BfNativeSampler' not in hlo and 'dot(' in hlo


def test_row_isolation_and_continuation(tf):
    cfg = core.Config(2, chains=2, warmup=0, draws=6, depth=4, adapt=False)
    target = lambda q, mean: -.5*tf.reduce_sum((q-mean)**2, -1)
    specs = [tf.TensorSpec([2], tf.float64)]
    batch = api.make_functions(tf, target, cfg, specs)
    scalar = api.make_functions(tf, target, replace(cfg, chains=1), specs)
    long = api.make_functions(tf, target, replace(cfg, draws=12), specs)
    q = tf.constant([[.1, .5], [4., -2.]], tf.float64)
    e = tf.constant([.2, .6], tf.float64); m = tf.ones_like(q)
    seed = tf.constant([[31, 0], [61, 0]], tf.int32); mean = tf.constant([.3, .8], tf.float64)
    a = batch.sample(q, e, m, seed, mean)
    for i in range(2):
        b = scalar.sample(q[i:i+1], e[i:i+1], m[i:i+1], seed[i:i+1], mean)
        assert bool(tf.reduce_all(tf.abs(a.draws[:, i]-b.draws[:, 0]) < 2e-12))
        assert bool(tf.reduce_all(a.seed[i] == b.seed[0]))
    b = batch.sample(a.position, e, m, a.seed, mean)
    both = long.sample(q, e, m, seed, mean)
    assert bool(tf.reduce_all(tf.concat((a.draws, b.draws), 0) == both.draws))
    assert bool(tf.reduce_all(b.seed == both.seed))


def test_signature_and_target_rejections(tf):
    cfg = core.Config(2, warmup=0, draws=1, adapt=False)
    with pytest.raises(ValueError):
        api.make_functions(tf, model(tf), cfg, [tf.TensorSpec([None, 2], tf.float64)])
    with pytest.raises(TypeError):
        api.make_functions(tf, model(tf), cfg, [tf.TensorSpec([], tf.string)])
    with pytest.raises(ValueError):
        api.make_functions(tf, model(tf), cfg, rng_layout='unknown')
    fn = api.make_functions(tf, lambda q: tf.reduce_sum(tf.cast(q, tf.float32), -1), cfg)
    with pytest.raises(TypeError):
        fn.value_gradient(tf.ones([1, 2], tf.float64))


def test_trace_graph_growth_is_linear_in_chains(tf):
    counts = []
    for c in (1, 4, 16):
        cfg = core.Config(2, chains=c, warmup=0, draws=1, adapt=False, depth=5)
        fn = api.make_functions(tf, lambda q: -.5*tf.reduce_sum(q*q, -1), cfg).sample
        graph = fn.get_concrete_function().graph.as_graph_def()
        counts.append(len(graph.node)+sum(len(f.node_def) for f in graph.library.function))
    assert counts[-1] < 16*counts[0]
