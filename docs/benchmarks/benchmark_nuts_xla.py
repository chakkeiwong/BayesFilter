"""Standalone owner-requested NUTS benchmark; no production tuning authority.

See docs/plans/bayesfilter-nuts-xla-benchmark-2026-10-27.md. All repeated
sampling/adaptation numerics are in enclosing XLA functions; host setup,
timing, serialization and post-run diagnostics are outside the timed region.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'gpu'), required=True)
    p.add_argument('--gpu', default='0', help='Physical GPU index for a GPU worker')
    p.add_argument('--target', choices=('gaussian', 'banana'), default='banana')
    p.add_argument('--method', choices=('nuts', 'hmc'), default='nuts')
    p.add_argument('--dimension', type=int, default=2)
    p.add_argument('--chains', type=int, default=4)
    p.add_argument('--warmup', type=int, default=128)
    p.add_argument('--draws', type=int, default=256)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--max-tree-depth', type=int, default=6)
    p.add_argument('--leapfrog-steps', type=int, default=8)
    p.add_argument('--step-size', type=float, default=.25)
    p.add_argument('--target-accept', type=float, default=.8)
    p.add_argument('--bend', type=float, default=.5)
    p.add_argument('--seed', type=int, default=20261027)
    p.add_argument('--threads', type=int, default=4)
    return p


def validate(args):
    for name in ('dimension', 'chains', 'warmup', 'draws', 'repeats',
                 'max_tree_depth', 'leapfrog_steps', 'threads'):
        if getattr(args, name) < 1:
            raise ValueError(f'{name} must be positive')
    if args.target == 'banana' and args.dimension % 2:
        raise ValueError('banana requires an even dimension')
    if args.chains < 2 or args.draws < 8 or args.warmup < 8:
        raise ValueError('diagnostic pilot needs >=2 chains and >=8 warmup/draws')
    if not math.isfinite(args.bend) or not math.isfinite(args.step_size) or args.step_size <= 0:
        raise ValueError('finite bend and positive finite step size required')
    if not 0 < args.target_accept < 1:
        raise ValueError('target acceptance must be in (0,1)')


def setup(args):
    # Configure before the first framework import. Every worker is a new process.
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if args.device == 'cpu' else args.gpu
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['BAYESFILTER_PRELOAD_CUSTOM_OP'] = '0'
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=args.device == 'gpu')
    import tensorflow_probability as tfp
    return tf, tfp, memory


def inverse_coordinates(tf, q, args):
    if args.target == 'gaussian':
        return q
    pairs = tf.reshape(q, [-1, args.dimension // 2, 2])
    x = pairs[..., 0]
    y = pairs[..., 1] - tf.cast(args.bend, q.dtype) * (tf.square(x) - 1.)
    return tf.reshape(tf.stack([x, y], axis=-1), tf.shape(q))


def target_log_prob(tf, args):
    def log_prob(q):
        z = inverse_coordinates(tf, q, args)
        return -.5 * tf.reduce_sum(tf.square(z), axis=-1)
    return log_prob


def make_chain(tf, tfp, args, *, adapting, log_prob_fn=None):
    count = args.warmup if adapting else args.draws
    log_prob = target_log_prob(tf, args) if log_prob_fn is None else log_prob_fn
    signature = [tf.TensorSpec([args.chains, args.dimension], tf.float64),
                 tf.TensorSpec([], tf.float64), tf.TensorSpec([2], tf.int32)]

    @tf.function(input_signature=signature, autograph=False, jit_compile=True)
    def chain(q, epsilon, seed):
        if args.method == 'nuts':
            kernel = tfp.mcmc.NoUTurnSampler(
                log_prob, epsilon, max_tree_depth=args.max_tree_depth,
                max_energy_diff=1000., unrolled_leapfrog_steps=1, parallel_iterations=1)
        else:
            kernel = tfp.mcmc.HamiltonianMonteCarlo(
                log_prob, epsilon, num_leapfrog_steps=args.leapfrog_steps)
        if adapting:
            kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
                kernel, num_adaptation_steps=max(1, args.warmup * 4 // 5),
                target_accept_prob=args.target_accept)
        kr = kernel.bootstrap_results(q)
        states = tf.TensorArray(tf.float64, size=count,
                               element_shape=[args.chains, args.dimension])
        telemetry = tf.TensorArray(tf.float64, size=count,
                                  element_shape=[args.chains, 7])

        def body(i, position, results, rng, saved, stats):
            keys = tf.random.experimental.stateless_split(rng, 2)
            position, results = kernel.one_step(position, results, seed=keys[1])
            base = results.inner_results if adapting else results
            cache = base if args.method == 'nuts' else base.accepted_results
            finite = tf.reduce_all(tf.math.is_finite(position), axis=-1)
            finite &= tf.math.is_finite(cache.target_log_prob)
            for grad in cache.grads_target_log_prob:
                finite &= tf.reduce_all(tf.math.is_finite(grad), axis=-1)
            log_accept = base.log_accept_ratio
            valid_accept = ~tf.math.is_nan(log_accept) & (log_accept < math.inf)
            accept = tf.exp(tf.minimum(log_accept, tf.constant(0., tf.float64)))
            if args.method == 'nuts':
                leaps = tf.cast(base.leapfrogs_taken, tf.float64)
                cap = tf.cast(base.reach_max_depth, tf.float64)
                divergence = tf.cast(base.has_divergence, tf.float64)
            else:
                leaps = tf.fill([args.chains], tf.cast(args.leapfrog_steps, tf.float64))
                cap = tf.zeros([args.chains], tf.float64)
                # HMC has no native NUTS divergence field; N/A is in metadata.
                divergence = tf.zeros([args.chains], tf.float64)
            row = tf.stack([accept, tf.cast(base.is_accepted, tf.float64),
                            leaps, cap, divergence, tf.cast(finite, tf.float64),
                            tf.cast(valid_accept, tf.float64)], axis=-1)
            return (i + 1, position, results, keys[0], saved.write(i, position),
                    stats.write(i, row))

        _, q, kr, _, states, telemetry = tf.while_loop(
            lambda i, *_: i < count, body,
            (tf.constant(0), q, kr, seed, states, telemetry), parallel_iterations=1)
        epsilon = kr.new_step_size if adapting else epsilon
        samples, stats = states.stack(), telemetry.stack()
        # Depends on both full outputs: synchronizes timing without bulk copies.
        token = tf.reduce_sum(samples) + tf.reduce_sum(stats)
        return q, epsilon, samples, stats, token
    return chain


def verify_graph(function):
    graph = function.get_concrete_function().graph.as_graph_def()
    nodes = list(graph.node)
    for fn in graph.library.function:
        nodes.extend(fn.node_def)
    forbidden = sorted({node.op for node in nodes
                        if 'PyFunc' in node.op or 'PFor' in node.op})
    if forbidden:
        raise RuntimeError(f'forbidden host callbacks/pfor: {forbidden}')
    return {'node_count_including_library': len(nodes), 'forbidden_ops': forbidden}


def timed(function, *inputs):
    start = time.perf_counter()
    result = function(*inputs)
    result[-1].numpy().item()
    return result, time.perf_counter() - start


def json_tensor(tf, value):
    if tf.is_tensor(value):
        return value.numpy().tolist()
    if isinstance(value, dict):
        return {k: json_tensor(tf, v) for k, v in value.items()}
    return value


def diagnostics(tf, samples, telemetry, args):
    from bayesfilter.inference.hmc_posterior_diagnostics import (
        rank_normalized_split_rhat, rank_normalized_bulk_tail_ess)
    chain_major = tf.transpose(samples, [1, 0, 2])
    flat = tf.reshape(samples, [-1, args.dimension])
    z = inverse_coordinates(tf, flat, args)
    rhat = rank_normalized_split_rhat(chain_major)
    ess = rank_normalized_bulk_tail_ess(chain_major)
    mean = tf.reduce_mean(flat, axis=0)
    second = tf.reduce_mean(tf.square(flat), axis=0)
    expected = [1. if args.target == 'gaussian' or j % 2 == 0
                else 1. + 2. * args.bend ** 2 for j in range(args.dimension)]
    out = {'rhat': rhat, 'ess': ess, 'mean': mean, 'second_moment': second,
           'expected_second_moment': expected,
           'inverse_mean': tf.reduce_mean(z, axis=0),
           'inverse_second_moment': tf.reduce_mean(tf.square(z), axis=0),
           'acceptance_probability': tf.reduce_mean(telemetry[..., 0]),
           'selected_state_fraction': tf.reduce_mean(telemetry[..., 1]),
           'reported_leapfrogs_mean': tf.reduce_mean(telemetry[..., 2]),
           'reported_leapfrogs_total': tf.reduce_sum(telemetry[..., 2]),
           'max_chain_leapfrogs_per_transition_mean':
               tf.reduce_mean(tf.reduce_max(telemetry[..., 2], axis=1)),
           'tree_cap_fraction': tf.reduce_mean(telemetry[..., 3]),
           'native_divergence_fraction': tf.reduce_mean(telemetry[..., 4])
               if args.method == 'nuts' else None,
           'finite_retained_cache': tf.reduce_all(telemetry[..., 5] == 1.),
           'valid_acceptance_telemetry': tf.reduce_all(telemetry[..., 6] == 1.)}
    if args.target == 'banana':
        pairs = tf.reshape(flat, [-1, args.dimension // 2, 2])
        out['x_squared_y_mean'] = tf.reduce_mean(
            tf.square(pairs[..., 0]) * pairs[..., 1], axis=0)
        out['expected_x_squared_y'] = 2. * args.bend
    return json_tensor(tf, out)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    validate(args)
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    config = vars(args).copy()
    config['output'] = str(config['output'])
    manifest = {'schema': 'bayesfilter.nuts_xla_benchmark.v1', 'config': config,
        'command': sys.argv, 'python': sys.executable, 'platform': platform.platform(),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'benchmark_sha256': source_hash(__file__),
        'plan': 'docs/plans/bayesfilter-nuts-xla-benchmark-2026-10-27.md',
        'result_note': 'docs/plans/bayesfilter-nuts-xla-benchmark-result-2026-10-27.md',
        'data_version': 'analytic_target_v1_no_external_data',
        'configuration_status': 'PILOT_HYPOTHESES_NOT_CANONICALLY_TUNED',
        'scientific_promotion_authority': False, 'jit_compile': True,
        'dtype': 'float64', 'tf32': False, 'mass_matrix': 'identity',
        'adaptation_fraction': '4/5 convenience; final 1/5 frozen',
        'numerical_loop': 'whole_warmup_and_retained_chain_XLA',
        'host_only': ['setup', 'I/O', 'timing', 'post_run_diagnostics'],
        'gpu_intentionally_hidden': args.device == 'cpu',
        'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'load_before': os.getloadavg(),
        'telemetry_columns': ['acceptance_probability', 'selected_state',
            'reported_leapfrogs', 'tree_cap', 'native_divergence',
            'finite_cache', 'valid_log_acceptance'],
        'status': 'running'}
    write_json(args.output / 'manifest.json', manifest)
    try:
        tf, tfp, memory = setup(args)
        manifest.update(tensorflow=tf.__version__, tensorflow_probability=tfp.__version__,
                        gpu_memory_policy=memory)
        import inspect
        nuts_source = inspect.getfile(tfp.mcmc.NoUTurnSampler)
        manifest['tfp_nuts_source'] = nuts_source
        manifest['tfp_nuts_sha256'] = source_hash(nuts_source)
        manifest['source_hashes'] = {str(path.relative_to(ROOT)): source_hash(path)
            for path in [ROOT / 'bayesfilter/runtime/gpu_memory_policy.py',
                         ROOT / 'bayesfilter/inference/hmc_posterior_diagnostics.py',
                         ROOT / 'bayesfilter/inference/hmc_diagnostic_math.py',
                         ROOT / 'bayesfilter/inference/hmc_ess.py']}
        manifest['tracked_dirty_paths'] = subprocess.check_output(
            ['git', 'diff', '--name-only', '--', __file__,
             *manifest['source_hashes']], cwd=ROOT, text=True).splitlines()
        device = '/CPU:0' if args.device == 'cpu' else '/GPU:0'
        with tf.device(device):
            warmup = make_chain(tf, tfp, args, adapting=True)
            sample = make_chain(tf, tfp, args, adapting=False)
            initial_epsilon = tf.constant(args.step_size, tf.float64)
            initial = tf.zeros([args.chains, args.dimension], tf.float64)
            key = tf.constant([args.seed, 0], tf.int32)
            # Execute before exporting HLO so first-call time includes compilation.
            first_warm, warm_first_seconds = timed(warmup, initial, initial_epsilon, key)
            first_sample, sample_first_seconds = timed(sample, first_warm[0], first_warm[1], key)
            manifest['compilation'] = {}
            for label, function, values in [('warmup', warmup, (initial, initial_epsilon, key)),
                                            ('sample', sample, (first_warm[0], first_warm[1], key))]:
                ir = function.experimental_get_compiler_ir(*values)(stage='optimized_hlo')
                (args.output / f'{label}.optimized_hlo.txt').write_text(ir)
                manifest['compilation'][label] = {**verify_graph(function),
                    'optimized_hlo_sha256': hashlib.sha256(ir.encode()).hexdigest(),
                    'traces': function.experimental_get_tracing_count()}
            for value in tf.nest.flatten(first_sample):
                if f'device:{args.device.upper()}:0' not in value.device:
                    raise RuntimeError(f'unexpected output device: {value.device}')
            manifest['output_device'] = first_sample[2].device
            manifest['first_call_seconds'] = {'warmup': warm_first_seconds,
                                             'sample': sample_first_seconds}
            manifest['first_calls_are_compile_plus_execution'] = True
            write_json(args.output / 'manifest.json', manifest)
            rows = []
            for repeat in range(args.repeats):
                key = tf.constant([args.seed, repeat + 1], tf.int32)
                # Deliberately dispersed starts; no exact-reference starts used for timing.
                initial = 2. * tf.random.stateless_normal(
                    [args.chains, args.dimension], key, dtype=tf.float64)
                warmed, warm_seconds = timed(warmup, initial, initial_epsilon, key)
                sample_key = tf.random.experimental.stateless_fold_in(key, 1729)
                sampled, sample_seconds = timed(sample, warmed[0], warmed[1], sample_key)
                if not bool(tf.reduce_all(warmed[3][..., 5:] == 1.).numpy().item()):
                    raise RuntimeError('invalid warm-up retained cache or telemetry')
                stats = diagnostics(tf, sampled[2], sampled[3], args)
                if not stats['finite_retained_cache'] or not stats['valid_acceptance_telemetry']:
                    raise RuntimeError('invalid retained cache or telemetry')
                stats.update(repeat=repeat, seed=[args.seed, repeat + 1],
                    step_size=float(warmed[1].numpy().item()),
                    warmup_seconds=warm_seconds, sample_seconds=sample_seconds,
                    raw_draws_per_second=args.draws * args.chains / sample_seconds,
                    min_bulk_ess_per_second=min(stats['ess']['bulk']) / sample_seconds,
                    min_tail_ess_per_second=min(stats['ess']['tail']) / sample_seconds)
                for label, tensors in [('warmup', warmed), ('sample', sampled)]:
                    for name, tensor in [('states', tensors[2]), ('telemetry', tensors[3])]:
                        with tf.device('/CPU:0'):
                            tf.io.write_file(str(args.output / f'{label}-{repeat:02d}-{name}.tensor'),
                                             tf.io.serialize_tensor(tensor))
                rows.append(stats)
                write_json(args.output / 'replicates.json', rows)
                print(json.dumps({'repeat': repeat, 'device': args.device,
                    'sample_seconds': sample_seconds, 'step_size': stats['step_size'],
                    'rhat_max': max(stats['rhat']['maximum'])}), flush=True)
            summary = {'configuration_status': manifest['configuration_status'],
                'interpretation': 'descriptive engineering pilot; no supported sampler ranking',
                'median_sample_seconds': statistics.median(row['sample_seconds'] for row in rows),
                'median_warmup_seconds': statistics.median(row['warmup_seconds'] for row in rows),
                'rhat_max': max(max(row['rhat']['maximum']) for row in rows),
                'all_retained_caches_finite': True, 'replicates': len(rows)}
            write_json(args.output / 'summary.json', summary)
            for name, function in [('warmup', warmup), ('sample', sample)]:
                if function.experimental_get_tracing_count() != 1:
                    raise RuntimeError(f'unexpected retracing in {name}')
            if args.device == 'gpu':
                manifest['allocator_bytes'] = tf.config.experimental.get_memory_info('GPU:0')
        manifest['status'] = 'completed'
    except BaseException as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        manifest['wall_seconds'] = time.perf_counter() - start
        manifest['load_after'] = os.getloadavg()
        write_json(args.output / 'manifest.json', manifest)


if __name__ == '__main__':
    run(parser().parse_args())
