"""Standalone, finite-grid NUTS versus fixed-HMC benchmark; no admission authority.

Numerical transitions come only from benchmark_nuts_xla.make_chain. See the
2026-10-27 nuts-fixed-hmc comparison plan for the predeclared evidence contract.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time

import benchmark_nuts_xla as benchmark

ROOT = benchmark.ROOT
LENGTHS = (1, 3, 5, 8, 9, 13, 18, 25)
ACCEPTANCE_TARGETS = (.65, .8, .9)
OBSERVABLES = ('x', 'y', 'x_squared', 'y_squared', 'x_squared_y')
EXPECTED = (0., 0., 1., 1.5, 1.)
SEED = 20261027


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--resume-selection-from', type=Path,
                        help='Reuse a frozen selection; never tune on heldout draws')
    return parser.parse_args()


def sign_test(ratios):
    positive = sum(value > 1. for value in ratios)
    negative = sum(value < 1. for value in ratios)
    count = positive + negative
    if not count:
        return 1.
    return min(1., 2. * sum(math.comb(count, i)
                           for i in range(min(positive, negative) + 1)) / 2 ** count)


def observable_tensor(tf, samples):
    x, y = samples[..., 0], samples[..., 1]
    return tf.stack((x, y, x*x, y*y, x*x*y), axis=-1)


def finite_minimum(values):
    if not all(value is not None and math.isfinite(value) for value in values):
        return None
    return min(values)


def serialize_missing(value, missing, path=''):
    """Preserve undefined diagnostics as null with their exact paths."""
    if isinstance(value, dict):
        return {k: serialize_missing(v, missing, path + '/' + k) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_missing(v, missing, path + '/' + str(i)) for i, v in enumerate(value)]
    if isinstance(value, float) and not math.isfinite(value):
        missing.append(path)
        return None
    return value


def health(tf, result):
    if not bool(tf.reduce_all(result[3][..., 5:] == 1.).numpy().item()):
        raise RuntimeError('Invalid retained state/cache or acceptance telemetry')
    if not bool(tf.reduce_all(tf.math.is_finite(result[2])).numpy().item()):
        raise RuntimeError('Nonfinite retained state')
    for tensor in tf.nest.flatten(result):
        if 'device:CPU:0' not in tensor.device:
            raise RuntimeError(f'Incorrect placement: {tensor.device}')


def assess(tf, result, args, seconds):
    from bayesfilter.inference.hmc_ess import stan_cross_chain_ess
    from bayesfilter.inference.hmc_posterior_diagnostics import rank_normalized_split_rhat
    health(tf, result)
    report = benchmark.diagnostics(tf, result[2], result[3], args)
    observables = observable_tensor(tf, result[2])
    ess = stan_cross_chain_ess(observables)
    rhat = rank_normalized_split_rhat(tf.transpose(observables, [1, 0, 2]))
    if not bool(tf.reduce_all(tf.math.is_finite(ess) & (ess > 0.)).numpy().item()):
        raise RuntimeError('Nonconstant finite observable ESS required')
    minimum = float(tf.reduce_min(ess).numpy().item())
    tail = finite_minimum(report['ess']['tail'])
    report.update(
        observable_mean=benchmark.json_tensor(tf, tf.reduce_mean(observables, axis=[0, 1])),
        observable_mean_ess=benchmark.json_tensor(tf, ess),
        observable_rhat=benchmark.json_tensor(tf, rhat),
        primary_minimum_ess=minimum, primary_ess_per_second=minimum / seconds,
        sample_seconds=seconds,
        raw_draws_per_second=args.chains * args.draws / seconds,
        min_bulk_ess_per_second=min(report['ess']['bulk']) / seconds,
        min_tail_ess_per_second=tail / seconds if tail is not None else None,
        movements_per_chain=benchmark.json_tensor(tf, tf.reduce_sum(tf.cast(
            tf.reduce_any(result[2][1:] != result[2][:-1], axis=-1), tf.int32), axis=0)))
    missing = []
    report = serialize_missing(report, missing)
    report['unavailable_diagnostics'] = missing
    return report


def save_tensors(tf, folder, label, result):
    for name, tensor in (('states', result[2]), ('telemetry', result[3])):
        with tf.device('/CPU:0'):
            tf.io.write_file(str(folder / f'{label}-{name}.tensor'), tf.io.serialize_tensor(tensor))


def compile_function(tf, function, initial, epsilon, seed, folder, name):
    result, seconds = benchmark.timed(function, initial, epsilon, seed)
    health(tf, result)
    hlo = function.experimental_get_compiler_ir(initial, epsilon, seed)(stage='optimized_hlo')
    (folder / (name + '.optimized_hlo.txt')).write_text(hlo)
    info = dict(benchmark.verify_graph(function),
        first_call_compile_plus_execution_seconds=seconds,
        optimized_hlo_sha256=hashlib.sha256(hlo.encode()).hexdigest(),
        tracing_count=function.experimental_get_tracing_count())
    if info['tracing_count'] != 1:
        raise RuntimeError('Unexpected retracing')
    return info


def main(args):
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    write = benchmark.write_json
    config = benchmark.parser().parse_args([
        '--output', str(args.output), '--device', 'cpu', '--target', 'banana',
        '--chains', '4', '--dimension', '2', '--warmup', '512', '--draws', '1024'])
    manifest = {
        'schema': 'bayesfilter.nuts_fixed_hmc_comparison.v1',
        'status': 'running', 'configuration_status': 'FINITE_GRID_BENCHMARK_NOT_CANONICAL_TUNING',
        'command': [sys.executable, *sys.argv], 'git_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'python': sys.executable, 'platform': platform.platform(),
        'plan': 'docs/plans/bayesfilter-nuts-fixed-hmc-comparison-2026-10-27.md',
        'result_note': 'docs/plans/bayesfilter-nuts-fixed-hmc-result-2026-10-27.md',
        'gpu_intentionally_hidden': True, 'device': '/CPU:0', 'dtype': 'float64',
        'jit_compile': True, 'tf32': False, 'scientific_promotion_authority': False,
        'target': 'banana_v1_bend_0.5_dimension_2', 'data_version': 'analytic_no_external_data',
        'chains': 4, 'mass': 'identity', 'hmc_lengths': LENGTHS,
        'acceptance_targets': ACCEPTANCE_TARGETS,
        'tuning': {'warmup': 512, 'retained': 1024, 'replicas': 3},
        'validation': {'frozen_warmup': 1024, 'retained': 4096, 'replicas': 12},
        'seed': SEED, 'tuning_replica_ids': [0, 1, 2],
        'heldout_replica_ids': list(range(1000, 1012)),
        'rng': 'stateless_split_into_start_warmup_sample_and_order_keys',
        'observable_names': OBSERVABLES, 'observable_expectation': EXPECTED,
        'primary_metric': 'minimum_five_observable_mean_ESS_per_retained_second',
        'cpu_affinity': sorted(os.sched_getaffinity(0)), 'load_before': os.getloadavg(),
        'compile_records': {}, 'source_sha256': {}}
    archive = args.output / 'source'
    archive.mkdir()
    sources = [Path(__file__), Path(benchmark.__file__),
        ROOT / 'bayesfilter/runtime/gpu_memory_policy.py',
        ROOT / 'bayesfilter/inference/hmc_posterior_diagnostics.py',
        ROOT / 'bayesfilter/inference/hmc_diagnostic_math.py',
        ROOT / 'bayesfilter/inference/hmc_ess.py']
    for source in sources:
        relative = str(source.resolve().relative_to(ROOT))
        manifest['source_sha256'][relative] = benchmark.source_hash(source)
        shutil.copyfile(source, archive / (relative.replace('/', '_') + '.txt'))
    manifest['source_status'] = subprocess.check_output(
        ['git', 'status', '--short', '--', *manifest['source_sha256']], cwd=ROOT, text=True)
    write(args.output / 'manifest.json', manifest)
    try:
        tf, tfp, memory = benchmark.setup(config)
        manifest.update(tensorflow=tf.__version__, tensorflow_probability=tfp.__version__,
                        gpu_memory_policy=memory)
        tfp_mcmc_dir = Path(inspect.getfile(tfp.mcmc)).parent
        for name, source_file in (('nuts', 'nuts.py'), ('hmc', 'hmc.py'),
                                 ('dual_averaging', 'dual_averaging_step_size_adaptation.py')):
            filename = str(tfp_mcmc_dir / source_file)
            manifest['source_sha256'][filename] = benchmark.source_hash(filename)
            shutil.copyfile(filename, archive / (name + '.py.txt'))
        from bayesfilter.inference.hmc_ess import stan_cross_chain_ess

        @tf.function(input_signature=[tf.TensorSpec([2], tf.int32)],
                     autograph=False, jit_compile=True)
        def initial_and_keys(seed):
            keys = tf.random.experimental.stateless_split(seed, 4)
            position = 2. * tf.random.stateless_normal([4, 2], keys[0], dtype=tf.float64)
            order = tf.random.stateless_uniform([], keys[3], dtype=tf.float64)
            return position, keys[1], keys[2], order

        def inputs(replica):
            return initial_and_keys(tf.constant([SEED, replica], tf.int32))

        compilation_dir = args.output / 'compiled'
        compilation_dir.mkdir()
        sample_functions = {}
        candidate_rows = []
        # Python only schedules independent configurations; each entire chain is XLA.
        for method in (() if args.resume_selection_from else ('hmc', 'nuts')):
            lengths = LENGTHS if method == 'hmc' else (0,)
            for length in lengths:
                candidate = copy.copy(config)
                candidate.method = method
                candidate.leapfrog_steps = length if length else 8
                sample = benchmark.make_chain(tf, tfp, candidate, adapting=False)
                sample_functions[(method, length)] = sample
                q, kw, ks, _ = inputs(-1)
                compile_name = f'tuning-{method}-L{length}-sample'
                manifest['compile_records'][compile_name] = compile_function(
                    tf, sample, q, tf.constant(.25, tf.float64), ks, compilation_dir, compile_name)
                for acceptance in ACCEPTANCE_TARGETS:
                    candidate.target_accept = acceptance
                    warm = benchmark.make_chain(tf, tfp, copy.copy(candidate), adapting=True)
                    name = f'{method}-L{length}-a{acceptance}'
                    folder = args.output / 'tuning' / name
                    folder.mkdir(parents=True)
                    compile_name = name + '-warmup'
                    manifest['compile_records'][compile_name] = compile_function(
                        tf, warm, q, tf.constant(.25, tf.float64), kw, compilation_dir, compile_name)
                    replicas = []
                    for replica in range(3):
                        position, warm_key, sample_key, _ = inputs(replica)
                        warmed, warm_seconds = benchmark.timed(
                            warm, position, tf.constant(.25, tf.float64), warm_key)
                        health(tf, warmed)
                        sampled, seconds = benchmark.timed(sample, warmed[0], warmed[1], sample_key)
                        report = assess(tf, sampled, candidate, seconds)
                        report.update(replica=replica, epsilon=float(warmed[1].numpy().item()),
                                      warmup_seconds=warm_seconds)
                        save_tensors(tf, folder, f'{replica}-warmup', warmed)
                        save_tensors(tf, folder, f'{replica}-sample', sampled)
                        replicas.append(report)
                    row = {'name': name, 'method': method, 'length': length,
                           'target_acceptance': acceptance, 'replicas': replicas,
                           'median_primary_ess_per_second': statistics.median(
                               r['primary_ess_per_second'] for r in replicas),
                           'frozen_epsilon': statistics.median(r['epsilon'] for r in replicas)}
                    write(folder / 'result.json', row)
                    candidate_rows.append(row)
                    write(args.output / 'tuning.json', candidate_rows)
                    write(args.output / 'manifest.json', manifest)
                    print(json.dumps({k: row[k] for k in
                        ('name', 'median_primary_ess_per_second', 'frozen_epsilon')}), flush=True)

        # Save the decision before any held-out sample is generated.
        if args.resume_selection_from:
            prior = args.resume_selection_from.resolve()
            selected = json.loads((prior / 'frozen-selection.json').read_text())
            candidate_rows = json.loads((prior / 'tuning.json').read_text())
            manifest['selection_provenance'] = {
                'source_attempt': str(prior),
                'frozen_selection_sha256': benchmark.source_hash(prior / 'frozen-selection.json'),
                'reason': 'serialization-only repair; same heldout seeds rerun, no retuning',
                'independent_additional_replicas': False}
            write(args.output / 'tuning.json', candidate_rows)
        else:
            selected = {}
            for method in ('hmc', 'nuts'):
                row = max((r for r in candidate_rows if r['method'] == method),
                          key=lambda r: r['median_primary_ess_per_second'])
                selected[method] = {k: v for k, v in row.items() if k != 'replicas'}
        write(args.output / 'frozen-selection.json', selected)
        print(json.dumps({'frozen_selection': selected}), flush=True)
        manifest['selection_seconds_including_setup_compile_diagnostics_io'] = time.perf_counter() - started

        functions = {}
        validation_config = copy.copy(config)
        validation_config.draws = 4096
        for method, selection in selected.items():
            candidate = copy.copy(validation_config)
            candidate.method = method
            candidate.leapfrog_steps = selection['length'] if selection['length'] else 8
            frozen_warm_config = copy.copy(candidate)
            frozen_warm_config.draws = 1024
            warm = benchmark.make_chain(tf, tfp, frozen_warm_config, adapting=False)
            sample = benchmark.make_chain(tf, tfp, candidate, adapting=False)
            epsilon = tf.constant(selection['frozen_epsilon'], tf.float64)
            position, kw, ks, _ = inputs(-2)
            for phase, function, key in [('warmup', warm, kw), ('sample', sample, ks)]:
                name = 'heldout-' + method + '-' + phase
                manifest['compile_records'][name] = compile_function(
                    tf, function, position, epsilon, key, compilation_dir, name)
            functions[method] = (warm, sample, epsilon, candidate)
        write(args.output / 'manifest.json', manifest)
        heldout = []
        for replica in range(1000, 1012):
            position, kw, ks, order = inputs(replica)
            methods = ('hmc', 'nuts') if float(order.numpy().item()) < .5 else ('nuts', 'hmc')
            paired = {'replica': replica, 'method_order': methods, 'methods': {}}
            for method in methods:
                warm, sample, epsilon, candidate = functions[method]
                folder = args.output / 'heldout' / str(replica) / method
                folder.mkdir(parents=True)
                warmed, warm_seconds = benchmark.timed(warm, position, epsilon, kw)
                health(tf, warmed)
                sampled, seconds = benchmark.timed(sample, warmed[0], epsilon, ks)
                report = assess(tf, sampled, candidate, seconds)
                report.update(frozen_epsilon=float(epsilon.numpy().item()), warmup_seconds=warm_seconds)
                save_tensors(tf, folder, 'warmup', warmed)
                save_tensors(tf, folder, 'sample', sampled)
                write(folder / 'result.json', report)
                paired['methods'][method] = report
            heldout.append(paired)
            write(args.output / 'heldout.json', heldout)
            print(json.dumps({'heldout_replica': replica, 'order': methods,
                'hmc_metric': paired['methods']['hmc']['primary_ess_per_second'],
                'nuts_metric': paired['methods']['nuts']['primary_ess_per_second']}), flush=True)

        ratios = [r['methods']['hmc']['primary_ess_per_second'] /
                  r['methods']['nuts']['primary_ess_per_second'] for r in heldout]
        log_ratios = tf.math.log(tf.constant(ratios, tf.float64))
        indices = tf.random.stateless_uniform([4096, 12], [SEED, 2000],
                                             minval=0, maxval=12, dtype=tf.int32)
        bootstrap = tf.exp(tf.reduce_mean(tf.gather(log_ratios, indices), axis=1))
        interval = tfp.stats.percentile(bootstrap, [2.5, 97.5], interpolation='linear')
        summary = {'selected': selected, 'paired_primary_ratios': ratios,
            'geometric_mean_primary_ratio_hmc_over_nuts':
                float(tf.exp(tf.reduce_mean(log_ratios)).numpy().item()),
            'paired_bootstrap_95pct_interval': benchmark.json_tensor(tf, interval),
            'two_sided_exact_sign_p': sign_test(ratios), 'methods': {}}
        for method in ('hmc', 'nuts'):
            rows = [r['methods'][method] for r in heldout]
            means = tf.constant([r['observable_mean'] for r in rows], tf.float64)
            pooled = tf.reduce_mean(means, axis=0)
            mcse = tf.sqrt(tf.reduce_sum(tf.square(means - pooled), axis=0) / (12. * 11.))
            z = (pooled - tf.constant(EXPECTED, tf.float64)) / mcse
            rhats = [v for r in rows for v in r['rhat']['maximum'] + r['observable_rhat']['maximum']]
            available_rhat = all(v is not None and math.isfinite(v) for v in rhats)
            maximum_rhat = max(rhats) if available_rhat else None
            tail_rates = [r['min_tail_ess_per_second'] for r in rows]
            summary['methods'][method] = {
                'median_sample_seconds': statistics.median(r['sample_seconds'] for r in rows),
                'median_warmup_seconds': statistics.median(r['warmup_seconds'] for r in rows),
                'median_primary_ess_per_second': statistics.median(r['primary_ess_per_second'] for r in rows),
                'median_min_bulk_ess_per_second': statistics.median(r['min_bulk_ess_per_second'] for r in rows),
                'median_min_tail_ess_per_second': statistics.median(tail_rates)
                    if all(v is not None for v in tail_rates) else None,
                'mean_leapfrogs': statistics.mean(r['reported_leapfrogs_mean'] for r in rows),
                'mean_acceptance_probability': statistics.mean(r['acceptance_probability'] for r in rows),
                'max_rhat_coordinates_and_observables': maximum_rhat,
                'pooled_observable_mean': benchmark.json_tensor(tf, pooled),
                'replicated_mean_mcse': benchmark.json_tensor(tf, mcse),
                'reference_residual_in_mcse': benchmark.json_tensor(tf, z),
                'moment_screen_pass': bool(tf.reduce_all(tf.abs(z) <= 4.).numpy().item()),
                'rhat_screen_pass': available_rhat and maximum_rhat <= 1.01,
                'unavailable_diagnostics_by_replica': {str(r['replica']):
                    r['methods'][method]['unavailable_diagnostics'] for r in heldout
                    if r['methods'][method]['unavailable_diagnostics']},
                'zero_movement_chains_by_replica': {str(r['replica']):
                    [i for i, count in enumerate(r['methods'][method]['movements_per_chain']) if count == 0]
                    for r in heldout if 0 in r['methods'][method]['movements_per_chain']},
                'max_tree_cap_fraction': max(r['tree_cap_fraction'] for r in rows),
                'native_divergence_fractions': [r['native_divergence_fraction'] for r in rows]}
        summary['validity_screens_pass'] = all(
            r['moment_screen_pass'] and r['rhat_screen_pass'] for r in summary['methods'].values())
        low, high = summary['paired_bootstrap_95pct_interval']
        summary['statistically_supported_within_target_ranking'] = (
            summary['validity_screens_pass'] and (low > 1. or high < 1.)
            and summary['two_sided_exact_sign_p'] < .05)
        summary['qualification'] = 'Conditional on frozen grid selections and this target/device/dtype; no global optimum or default claim'
        write(args.output / 'summary.json', summary)
        for key, (warm, sample, _, _) in functions.items():
            if warm.experimental_get_tracing_count() != 1 or sample.experimental_get_tracing_count() != 1:
                raise RuntimeError(f'Unexpected heldout retracing: {key}')
        for function in sample_functions.values():
            if function.experimental_get_tracing_count() != 1:
                raise RuntimeError('Unexpected tuning sample retracing')
        manifest['status'] = 'completed'
        print(json.dumps({'status': 'completed', 'ratio': summary['geometric_mean_primary_ratio_hmc_over_nuts'],
                          'interval': summary['paired_bootstrap_95pct_interval'],
                          'screens_pass': summary['validity_screens_pass']}), flush=True)
    except BaseException as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        manifest['wall_seconds'] = time.perf_counter() - started
        manifest['load_after'] = os.getloadavg()
        write(args.output / 'manifest.json', manifest)


if __name__ == '__main__':
    main(arguments())
