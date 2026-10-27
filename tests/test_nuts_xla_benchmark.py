"""Independent target checks and whole-chain CPU/XLA mechanics smoke."""
import importlib.util
from pathlib import Path

import pytest


def benchmark():
    path = Path(__file__).resolve().parents[1] / 'docs/benchmarks/benchmark_nuts_xla.py'
    spec = importlib.util.spec_from_file_location('nuts_xla_benchmark', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_banana_dimension_fails_before_framework_setup(tmp_path):
    module = benchmark()
    args = module.parser().parse_args(['--output', str(tmp_path / 'new'),
                                     '--device', 'cpu', '--dimension', '3'])
    with pytest.raises(ValueError, match='even dimension'):
        module.run(args)
    assert not args.output.exists()


def test_banana_density_gradient_and_compiled_chains(tmp_path):
    module = benchmark()
    args = module.parser().parse_args(['--output', str(tmp_path / 'new'),
        '--device', 'cpu', '--warmup', '16', '--draws', '16'])
    tf, tfp, _ = module.setup(args)
    log_prob = module.target_log_prob(tf, args)

    @tf.function(input_signature=[tf.TensorSpec([4, 2], tf.float64)],
                 autograph=False, jit_compile=True)
    def score(q):
        with tf.GradientTape() as tape:
            tape.watch(q)
            value = log_prob(q)
        return value, tape.gradient(value, q)

    q = tf.constant([[0., 0.], [1., 2.], [-2., 1.], [.5, -.25]], tf.float64)
    values, gradients = score(q)
    for row, actual_value, actual_score in zip(q.numpy().tolist(),
            values.numpy().tolist(), gradients.numpy().tolist()):
        x, y = row
        residual = y - args.bend * (x * x - 1.)
        assert actual_value == pytest.approx(-.5 * (x * x + residual * residual))
        assert actual_score == pytest.approx([-x + 2. * args.bend * x * residual, -residual])

    with tf.device('/CPU:0'):
        warm = module.make_chain(tf, tfp, args, adapting=True)
        sample = module.make_chain(tf, tfp, args, adapting=False)
        epsilon = tf.constant(.25, tf.float64)
        seed = tf.constant([20261027, 1], tf.int32)
        warm_result, _ = module.timed(warm, q, epsilon, seed)
        result, _ = module.timed(sample, warm_result[0], warm_result[1], seed)
        same, _ = module.timed(sample, warm_result[0], warm_result[1], seed)
        other, _ = module.timed(sample, warm_result[0], warm_result[1], seed + [0, 1])
        assert result[2].shape == (16, 4, 2)
        assert bool(tf.reduce_all(result[2] == same[2]).numpy().item())
        assert not bool(tf.reduce_all(result[2] == other[2]).numpy().item())
        assert bool(tf.reduce_all(result[3][..., 5:] == 1.).numpy().item())
        assert warm.experimental_get_tracing_count() == sample.experimental_get_tracing_count() == 1
        for function, inputs in [(warm, (q, epsilon, seed)),
                                 (sample, (warm_result[0], warm_result[1], seed))]:
            assert not module.verify_graph(function)['forbidden_ops']
            assert 'HloModule' in function.experimental_get_compiler_ir(*inputs)(stage='optimized_hlo')
