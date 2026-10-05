"""Machine-local diagnostics for chain batching/process layouts.

Run ``python -m bayesfilter.inference.chain_benchmark --help``. These timings
compare execution layouts, not HMC kernels, ESS, posterior quality or tuning.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from bayesfilter.inference.chain_execution import (
    HMCChainExecutor,
    HMCChainKernel,
    HMCChainLayout,
    HMCChainTarget,
    _json_write,
    _positive,
    _states,
)


def _target_record(result):
    return [(value, score) for shard in result.shards
            for value, score in zip(shard['values'], shard['scores'], strict=True)]


def _compare_target(actual, expected, atol, rtol):
    if len(actual) != len(expected):
        raise ValueError('target comparison has a different number of chains')
    maximum = 0.
    for (av, ag), (ev, eg) in zip(actual, expected, strict=True):
        for a, e in zip((av, *ag), (ev, *eg), strict=True):
            maximum = max(maximum, abs(a - e))
            if not math.isfinite(a) or not math.isfinite(e) or abs(a - e) > atol + rtol * abs(e):
                raise ValueError('batched target values/scores disagree with singleton evaluation')
    return maximum


def benchmark_hmc_chain_execution(target, initial_state, kernel, layouts, *, output_dir,
                                  repetitions=3, warm_calls=5, timeout_seconds=300.,
                                  atol=1e-10, rtol=1e-10, max_energy_error=100.):
    """Measure native target evaluations and fixed HMC chunks on this machine.

    Each repetition uses fresh persistent workers; warm calls reuse their graphs.
    Layout order rotates. Numerical target parity against singleton batches is
    required. Failed cells are retained and excluded from fastest-observed advice.
    A caller may supply any importable filtering-backed adapter factory.
    """
    _positive(repetitions, 'repetitions')
    _positive(warm_calls, 'warm_calls')
    initial_state = _states(initial_state)
    layouts = tuple(layouts)
    if not layouts or any(not isinstance(layout, HMCChainLayout) for layout in layouts):
        raise ValueError('provide at least one HMCChainLayout')
    if any(not math.isfinite(x) or x < 0 for x in (atol, rtol)):
        raise ValueError('comparison tolerances must be finite and nonnegative')
    if not math.isfinite(max_energy_error) or max_energy_error <= 0:
        raise ValueError('max_energy_error must be finite and positive')
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True,
                                         stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    report = {'schema': 'bayesfilter.hmc_chain_benchmark.v1', 'status': 'running',
              'target': asdict(target), 'kernel': asdict(kernel),
              'layouts': [asdict(layout) for layout in layouts], 'chain_count': len(initial_state),
              'initial_state': initial_state, 'repetitions': repetitions, 'warm_calls': warm_calls,
              'command': sys.argv, 'python': sys.executable, 'platform': platform.platform(),
              'git_commit': commit, 'atol': atol, 'rtol': rtol, 'max_energy_error': max_energy_error,
              'cells': [], 'nonclaims': ['Fixed-kernel hardware diagnostic, not posterior samples or tuning authority.',
                  'Fastest observed is descriptive; no statistical superiority or universal recommendation.',
                  'Changing chain batch size changes random streams; worker count alone does not.',
                  'First-call time includes tracing, compilation and execution; it is not compiler-only time.',
                  'Host RSS and TF live/peak allocator bytes are distinct from GPU process reservation.']}
    _json_write(output / 'manifest.json', report)
    began = time.perf_counter()
    reference_layout = HMCChainLayout(1, 1, 1, layouts[0].device,
                                     layouts[0].gpu_devices[:1], layouts[0].pin_cpu, layouts[0].tf32)
    try:
        with HMCChainExecutor(target, initial_state, kernel, reference_layout,
                              output_dir=output / 'singleton-reference', timeout_seconds=timeout_seconds) as executor:
            reference = _target_record(executor.value_and_score())
            identity = executor.startup[0]['target_signature']
            factory_hash = executor.startup[0]['factory_sha256']
            execution_hashes = executor.startup[0]['execution_source_sha256']
    except Exception as error:  # noqa: BLE001 -- preserve arbitrary application factory failures
        report.update(status='reference_failed', error=repr(error), summaries=[],
                      elapsed_seconds=time.perf_counter() - began, fastest_observed_layout=None)
        _json_write(output / 'result.json', report)
        (output / 'report.md').write_text(
            '# HMC chain execution benchmark\n\nSingleton reference failed; no layout is eligible.\n'
            'See result.json and singleton-reference worker logs.\n')
        return report
    report['reference_target_signature'] = identity
    report['reference_factory_sha256'] = factory_hash
    report['reference_execution_source_sha256'] = execution_hashes
    for repetition in range(repetitions):
        order = [(i + repetition) % len(layouts) for i in range(len(layouts))]
        for index in order:
            cell_dir = output / f'repetition-{repetition:02d}-layout-{index:02d}'
            cell = {'repetition': repetition, 'layout': index, 'path': str(cell_dir)}
            try:
                with HMCChainExecutor(target, initial_state, kernel, layouts[index],
                                      output_dir=cell_dir, timeout_seconds=timeout_seconds) as executor:
                    if any(r['target_signature'] != identity or r['factory_sha256'] != factory_hash
                           or r['execution_source_sha256'] != execution_hashes
                           for r in executor.startup):
                        raise ValueError('target or factory identity differs across layouts')
                    first_target = executor.value_and_score()
                    error = _compare_target(_target_record(first_target), reference, atol, rtol)
                    target_times = []
                    for _ in range(warm_calls):
                        target_call = executor.value_and_score()
                        error = max(error, _compare_target(_target_record(target_call), reference, atol, rtol))
                        if any(s['trace_count'] != 1 for s in target_call.shards):
                            raise ValueError('benchmark retraced target evaluation')
                        target_times.append(target_call.metadata['end_to_end_seconds'])
                    first = executor.run(current_state=initial_state, seed=kernel.seed)
                    calls = [first]
                    calls += [executor.run(current_state=initial_state, seed=kernel.seed)
                              for _ in range(warm_calls)]
                    if any(not all(s['health']['chain_moved']) for call in calls for s in call.shards):
                        raise ValueError('a benchmark chain did not move; timing is not eligible')
                    if any(s['health'].get('maximum_absolute_log_accept_ratio', math.inf) > max_energy_error
                           for call in calls for s in call.shards):
                        raise ValueError('HMC energy-error veto; timing is not eligible')
                    cold_hashes = [s['tensors']['samples']['sha256'] for s in first.shards]
                    if any([s['tensors']['samples']['sha256'] for s in call.shards] != cold_hashes
                           for call in calls[1:]):
                        raise ValueError('fixed-state/seed replay changed during benchmark')
                    if any(s['trace_count'] != 1 for call in calls for s in call.shards):
                        raise ValueError('benchmark retraced a fixed numerical configuration')
                    cell.update(status='passed', startup_seconds=executor.startup_seconds,
                        target_first_seconds=first_target.metadata['end_to_end_seconds'],
                        target_warm_seconds=target_times, maximum_value_score_error=error,
                        hmc_first_seconds=first.metadata['end_to_end_seconds'],
                        hmc_warm_seconds=[c.metadata['end_to_end_seconds'] for c in calls[1:]],
                        hmc_worker_call_seconds=[[sum(s['call_seconds'] for s in r['shards'])
                                                for r in c.metadata['workers']] for c in calls],
                        memory=[[r['memory'] for r in c.metadata['workers']] for c in calls],
                        worker_provenance=executor.startup,
                        health=[s['health'] for s in first.shards])
            except Exception as error:  # noqa: BLE001 -- failed cells remain reviewable and ineligible
                cell.update(status='failed', error=repr(error))
            report['cells'].append(cell)
            _json_write(output / f'cell-{repetition:02d}-{index:02d}.json', cell)
    summaries = []
    for index, layout in enumerate(layouts):
        cells = [c for c in report['cells'] if c['layout'] == index]
        valid = len(cells) == repetitions and all(c['status'] == 'passed' for c in cells)
        row = {'layout': index, 'configuration': asdict(layout), 'eligible': valid}
        if valid:
            medians = [statistics.median(c['hmc_warm_seconds']) for c in cells]
            warm = statistics.median(medians)
            row.update(hmc_warm_median_seconds=warm, process_repetition_medians=medians,
                       hmc_warm_min_seconds=min(medians), hmc_warm_max_seconds=max(medians),
                       transitions_per_second=len(initial_state) * kernel.num_results / warm,
                       target_warm_median_seconds=statistics.median(
                           statistics.median(c['target_warm_seconds']) for c in cells),
                       startup_median_seconds=statistics.median(c['startup_seconds'] for c in cells),
                       hmc_first_median_seconds=statistics.median(c['hmc_first_seconds'] for c in cells),
                       host_peak_rss_sum_max_bytes=max(sum(w['host_peak_rss_bytes'] for w in call)
                                                     for c in cells for call in c['memory']),
                       gpu_allocator_peak_sum_max_bytes=(
                           max(sum(w['gpu_allocator']['peak'] for w in call)
                               for c in cells for call in c['memory']) if layout.device == 'GPU' else None))
        summaries.append(row)
    eligible = [r for r in summaries if r['eligible']]
    report.update(status='completed' if len(eligible) == len(layouts) else 'completed_with_failed_layouts',
                  summaries=summaries, elapsed_seconds=time.perf_counter() - began,
                  fastest_observed_layout=min(eligible, key=lambda r: r['hmc_warm_median_seconds'])['layout']
                  if eligible else None)
    _json_write(output / 'result.json', report)
    lines = ['# HMC chain execution benchmark', '',
             'Hardware diagnostic only. Fastest observed is descriptive, not a tuning or convergence decision.', '',
             '| Layout | Batch / workers / cores per worker | Startup s | First HMC s | Warm HMC s | Transitions/s | Host peak sum MiB |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in summaries:
        cfg = row['configuration']
        label = f"{cfg['chains_per_batch']} / {cfg['workers']} / {cfg['cores_per_worker']}"
        values = (f"{row['startup_median_seconds']:.3f} | {row['hmc_first_median_seconds']:.3f} | "
                  f"{row['hmc_warm_median_seconds']:.6f} | {row['transitions_per_second']:.3f} | "
                  f"{row['host_peak_rss_sum_max_bytes'] / 1024**2:.1f}" if row['eligible']
                  else 'failed | ineligible | — | — | —')
        lines.append(f"| {row['layout']} | {label} | {values} |")
    lines += ['', 'Timings include synchronization, IPC and tensor artifacts. Host peak sums are sums of worker',
              'lifetime high-water marks, not simultaneous live memory. GPU allocator peaks are reset per call.',
              'Target timings, allocator memory, parity, health, failures and provenance are in result.json.',
              'An eligible short timing run does not establish convergence, posterior validity or tuning authority.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target-factory', default='bayesfilter.testing.hmc_chain_benchmark_targets:kalman_target')
    parser.add_argument('--target-kwargs', type=Path, help='JSON keyword arguments for your target factory')
    parser.add_argument('--initial-state', type=Path, help='JSON [chains, parameters]; required for custom factories')
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--layout', action='append', help='BATCH:WORKERS:CORES; repeat to compare layouts')
    parser.add_argument('--device', choices=('GPU', 'CPU'), default='GPU')
    parser.add_argument('--gpu', action='append', default=[], help='One shared GPU ID or one per worker')
    parser.add_argument('--draws', type=int, default=16)
    parser.add_argument('--leapfrog-steps', type=int, default=3)
    parser.add_argument('--step-size', type=float, default=.03)
    parser.add_argument('--dtype', choices=('float32', 'float64'), default='float64')
    parser.add_argument('--seed', type=int, nargs=2, default=(20261001, 17))
    parser.add_argument('--target-scope')
    parser.add_argument('--target-status', choices=('auto', 'none', 'per_chain_step'), default='auto')
    parser.add_argument('--no-jit-compile', action='store_true', help='Explicit diagnostic exception')
    parser.add_argument('--non-xla-reason', help='Required with --no-jit-compile')
    parser.add_argument('--atol', type=float, default=1e-10)
    parser.add_argument('--rtol', type=float, default=1e-10)
    parser.add_argument('--max-energy-error', type=float, default=100.)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--warm-calls', type=int, default=5)
    parser.add_argument('--timeout-seconds', type=float, default=300.)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.no_jit_compile and not (args.non_xla_reason or '').strip():
        parser.error('--no-jit-compile requires --non-xla-reason')
    if args.initial_state:
        initial = json.loads(args.initial_state.read_text())
    else:
        if args.target_factory != 'bayesfilter.testing.hmc_chain_benchmark_targets:kalman_target':
            parser.error('--initial-state is required for custom target factories')
        initial = [[.2 + i * .01, -1.05 + i * .01] for i in range(args.chains)]
    target = HMCChainTarget(args.target_factory,
                           json.loads(args.target_kwargs.read_text()) if args.target_kwargs else {})
    target_scope = args.target_scope
    if target_scope is None and args.target_factory.startswith('bayesfilter.testing.hmc_chain_benchmark_targets:'):
        target_scope = 'hmc-chain-benchmark-fixture-v1'
    layouts = []
    chains = len(initial)
    for spec in args.layout or [f'{chains}:1:{chains}', f'1:{chains}:1']:
        batch, workers, cores = (int(x) for x in spec.split(':'))
        layouts.append(HMCChainLayout(batch, workers, cores, args.device, tuple(args.gpu)))
    result = benchmark_hmc_chain_execution(target, initial,
        HMCChainKernel(args.draws, args.step_size, args.leapfrog_steps, seed=tuple(args.seed),
                       dtype=args.dtype, jit_compile=not args.no_jit_compile,
                       non_xla_reason=args.non_xla_reason, target_scope=target_scope,
                       target_status_trace_policy=args.target_status), layouts,
        output_dir=args.output_dir, repetitions=args.repetitions, warm_calls=args.warm_calls,
        timeout_seconds=args.timeout_seconds, atol=args.atol, rtol=args.rtol,
        max_energy_error=args.max_energy_error)
    print(json.dumps({'status': result['status'], 'fastest_observed_layout': result['fastest_observed_layout'],
                      'report': str(args.output_dir / 'report.md')}))
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
