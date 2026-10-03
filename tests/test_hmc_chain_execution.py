"""Bounded CPU/XLA engineering checks; no posterior or tuning qualification."""

import json
import os
import signal
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from bayesfilter.inference.chain_benchmark import benchmark_hmc_chain_execution
from bayesfilter.inference.chain_execution import (
    HMCChainExecutionError,
    HMCChainExecutor,
    HMCChainKernel,
    HMCChainLayout,
    HMCChainTarget,
)

GAUSSIAN = HMCChainTarget('bayesfilter.testing.hmc_chain_benchmark_targets:gaussian_target')
INITIAL = [[.2, -.5], [.4, .7], [-.6, .1]]
KERNEL = HMCChainKernel(6, .12, 2, target_scope='hmc-chain-benchmark-fixture-v1')


def _hashes(result):
    return [s['tensors']['samples']['sha256'] for s in result.shards]


def _closed(executor):
    assert executor._closed
    assert all(p.poll() is not None for p in executor._processes)
    assert all(p.stdout.closed and p.stdin.closed for p in executor._processes)


def status_target(invalid=False):
    import tensorflow as tf

    from bayesfilter.testing.hmc_chain_benchmark_targets import GaussianBenchmarkTarget

    class StatusTarget(GaussianBenchmarkTarget):
        def target_status_telemetry(self, state):
            squared = tf.reduce_sum(state * state, axis=-1)
            valid = squared < (1. if invalid else 1e10)
            return {'status_code': tf.where(valid, 0, 1),
                    'valid_pre_regularized_score': valid,
                    'floor_count_value': tf.zeros_like(squared, tf.int32),
                    'diagnostic_limits': 'Non-numerical notes stay outside compiled outputs.'}

    return StatusTarget()


def coupled_target():
    import tensorflow as tf

    from bayesfilter.testing.hmc_chain_benchmark_targets import GaussianBenchmarkTarget

    class Coupled(GaussianBenchmarkTarget):
        def log_prob_and_grad(self, state):
            value, score = super().log_prob_and_grad(state)
            return value + tf.reduce_sum(state), score

    return Coupled()


def wrong_dtype_target():
    import tensorflow as tf

    from bayesfilter.testing.hmc_chain_benchmark_targets import GaussianBenchmarkTarget

    class WrongDtype(GaussianBenchmarkTarget):
        def log_prob_and_grad(self, state):
            return super().log_prob_and_grad(tf.cast(state, tf.float64))

    return WrongDtype()


def test_defaults_configuration_and_lazy_public_import(tmp_path):
    assert HMCChainLayout(4).device == 'GPU'
    assert KERNEL.jit_compile
    assert KERNEL.target_status_trace_policy == 'auto'
    assert HMCChainLayout.one_process_per_chain(2, cores_per_chain=3).cores_per_worker == 3
    with pytest.raises(ValueError, match='diagnostic'):
        replace(KERNEL, jit_compile=False)
    with pytest.raises(ValueError, match='int32'):
        replace(KERNEL, seed=(2**40, 0))
    with pytest.raises(ValueError, match='finite'):
        replace(KERNEL, step_size=float('nan'))
    with pytest.raises(ValueError, match='workers'):
        HMCChainLayout(4, 2).partition(4)
    with pytest.raises(ValueError, match='core budgets'):
        HMCChainExecutor(GAUSSIAN, INITIAL, KERNEL,
                         HMCChainLayout(3, cores_per_worker=len(os.sched_getaffinity(0)) + 1, device='CPU'))
    with pytest.raises(ValueError, match='finite'):
        HMCChainExecutor(GAUSSIAN, [[float('nan')]], KERNEL, HMCChainLayout(1, device='CPU'))
    env = dict(os.environ, BAYESFILTER_PRELOAD_CUSTOM_OP='0', CUDA_VISIBLE_DEVICES='-1')
    result = subprocess.run([sys.executable, '-c',
        ('import sys; from bayesfilter.inference import HMCChainExecutor, benchmark_hmc_chain_execution; '
         'assert "tensorflow" not in sys.modules')], env=env, capture_output=True,
        text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr


def test_automatic_gpu_selection_does_not_take_display_fallback(tmp_path, monkeypatch):
    from bayesfilter.runtime import display_gpu_policy

    monkeypatch.setattr(display_gpu_policy, 'probe_inventory', dict)
    monkeypatch.setattr(display_gpu_policy, 'select_gpu',
                        lambda _: {'selected': {'uuid': 'GPU-display', 'is_display': True}})
    with pytest.raises(HMCChainExecutionError, match='non-display'):
        HMCChainExecutor(GAUSSIAN, INITIAL, KERNEL, HMCChainLayout(3),
                         output_dir=tmp_path / 'no-display')
    assert (tmp_path / 'no-display' / 'startup-failure.json').exists()


def test_native_batches_process_partition_replay_and_dynamic_inputs(tmp_path):
    records = []
    for workers in (1, 2):
        with HMCChainExecutor(GAUSSIAN, INITIAL, KERNEL,
                             HMCChainLayout(2, workers, 1, 'CPU'),
                             output_dir=tmp_path / f'workers-{workers}') as executor:
            values = executor.value_and_score()
            actual = [value for s in values.shards for value in s['values']]
            expected = [-.5 * sum(x*x for x in row) for row in INITIAL]
            assert actual == pytest.approx(expected, abs=1e-14)
            assert all(s['xla_must_compile'] for s in values.shards)
            first = executor.run(current_state=INITIAL, seed=KERNEL.seed)
            repeat = executor.run(current_state=INITIAL, seed=KERNEL.seed)
            assert _hashes(first) == _hashes(repeat)
            assert all(s['trace_count'] == 1 and s['xla_must_compile'] for s in repeat.shards)
            assert [(s['start'], s['stop']) for s in repeat.shards] == [(0, 2), (2, 3)]
            assert all(r['intra_op_threads'] == 1 and len(r['cpu_affinity']) == 1 for r in executor.startup)
            assert len({tuple(r['cpu_affinity']) for r in executor.startup}) == workers
            records.append(first)
            if workers == 1:
                assert _hashes(executor.run(current_state=INITIAL, seed=(9, 2))) != _hashes(first)
                assert _hashes(executor.run(current_state=INITIAL, seed=KERNEL.seed, step_size=.2)) != _hashes(first)
                changed_state = [[x + .25 for x in row] for row in INITIAL]
                shifted = executor.value_and_score(changed_state)
                assert shifted.shards[0]['values'] != values.shards[0]['values']
                assert all(s['trace_count'] == 1 for s in shifted.shards)
                assert _hashes(executor.run(current_state=changed_state, seed=KERNEL.seed)) != _hashes(first)
                continued = executor.run(seed=(8, 4))
                assert _hashes(continued) != _hashes(first)
                with pytest.raises(ValueError, match='shape'):
                    executor.run(current_state=[[0.]])
        _closed(executor)
    assert _hashes(records[0]) == _hashes(records[1])
    import tensorflow as tf

    from bayesfilter.inference.hmc import FullChainHMCConfig, ReusableFullChainHMCRunner
    from bayesfilter.testing.hmc_chain_benchmark_targets import gaussian_target

    samples = records[0].load()
    assert samples.shape == (KERNEL.num_results, 3, 2)
    for shard in records[0].shards:
        direct = ReusableFullChainHMCRunner(gaussian_target(),
            tf.constant(INITIAL[shard['start']:shard['stop']], tf.float64),
            FullChainHMCConfig(num_results=KERNEL.num_results, num_burnin_steps=0,
                step_size=KERNEL.step_size, num_leapfrog_steps=KERNEL.num_leapfrog_steps,
                seed=KERNEL.seed, use_xla=True, capture_candidate_health=True,
                target_scope=KERNEL.target_scope)).run(seed=shard['seed'])
        tf.debugging.assert_equal(samples[:, shard['start']:shard['stop']], direct.samples)
    # Loading verifies that the returned shards still contain the recorded data.
    Path(records[1].shards[0]['tensors']['samples']['path']).write_bytes(b'changed')
    with pytest.raises(HMCChainExecutionError, match='changed tensor'):
        records[1].load()


def test_float32_and_one_compilation_for_equal_batches(tmp_path):
    kernel = replace(KERNEL, dtype='float32')
    with HMCChainExecutor(GAUSSIAN, INITIAL, kernel, HMCChainLayout(1, device='CPU'),
                         output_dir=tmp_path / 'float32') as executor:
        value = executor.value_and_score()
        result = executor.run()
        assert all(s['tensors']['value']['dtype'] == 'float32' for s in value.shards)
        assert all(s['tensors']['samples']['dtype'] == 'float32' for s in result.shards)
        assert all(s['trace_count'] == 1 for s in result.shards)
    _closed(executor)


def test_hmc_call_alone_rejects_wrong_target_dtype(tmp_path):
    with (HMCChainExecutor(HMCChainTarget('tests.test_hmc_chain_execution:wrong_dtype_target'),
                         INITIAL, replace(KERNEL, dtype='float32'), HMCChainLayout(3, device='CPU'),
                         output_dir=tmp_path / 'dtype-mismatch') as executor,
          pytest.raises(HMCChainExecutionError, match='preserve the configured dtype')):
        executor.run()
    _closed(executor)


def test_non_xla_is_explicit_and_reported(tmp_path):
    kernel = replace(KERNEL, jit_compile=False, non_xla_reason='tiny graph reference check')
    with HMCChainExecutor(GAUSSIAN, INITIAL, kernel, HMCChainLayout(3, device='CPU'),
                         output_dir=tmp_path / 'graph-reference') as executor:
        result = executor.run()
        assert not result.shards[0]['xla_must_compile']
        assert not result.metadata['kernel']['jit_compile']
        assert result.metadata['kernel']['non_xla_reason']


def test_status_auto_and_invalid_rejected_proposal(tmp_path):
    target = HMCChainTarget('tests.test_hmc_chain_execution:status_target', {'invalid': True})
    with HMCChainExecutor(target, [[0., 0.], [0., 0.]], replace(KERNEL, step_size=12.),
                         HMCChainLayout(2, device='CPU'), output_dir=tmp_path / 'status') as executor:
        assert executor.startup[0]['target_status_trace_policy'] == 'per_chain_step'
        assert executor.value_and_score().metadata['all_healthy']
        with pytest.raises(HMCChainExecutionError, match='invalid target or HMC health'):
            executor.run()
    _closed(executor)
    call = json.loads((tmp_path / 'status' / 'call-00002.json').read_text())
    shard = call['workers'][0]['shards'][0]
    assert shard['health']['status']['target_status_telemetry']['valid']
    assert not shard['health']['status']['proposed_target_status_telemetry']['valid']
    assert Path(shard['trace']['proposed_state']['path']).exists()


@pytest.mark.parametrize('failure', ['exit', 'deadline'])
def test_failed_worker_closes_every_process(tmp_path, failure):
    executor = HMCChainExecutor(GAUSSIAN, INITIAL[:2], KERNEL,
        HMCChainLayout.one_process_per_chain(2, device='CPU'), output_dir=tmp_path / failure)
    executor.timeout_seconds = .2
    os.kill(executor._processes[0].pid, signal.SIGKILL if failure == 'exit' else signal.SIGSTOP)
    with pytest.raises((HMCChainExecutionError, BrokenPipeError)):
        executor.run()
    _closed(executor)
    assert list((tmp_path / failure).glob('failure-*.json'))


def test_reference_failure_and_invalid_layout_recorded(tmp_path):
    bad = HMCChainTarget('bayesfilter.testing.hmc_chain_benchmark_targets:missing_factory')
    failed = benchmark_hmc_chain_execution(bad, INITIAL, KERNEL, [HMCChainLayout(3, device='CPU')],
        output_dir=tmp_path / 'reference-failure', repetitions=1, warm_calls=1)
    assert failed['status'] == 'reference_failed'
    assert failed['fastest_observed_layout'] is None
    assert (tmp_path / 'reference-failure' / 'result.json').exists()
    report = benchmark_hmc_chain_execution(GAUSSIAN, INITIAL, KERNEL,
        [HMCChainLayout(3, device='CPU'), HMCChainLayout(3, 2, device='CPU')],
        output_dir=tmp_path / 'benchmark', repetitions=1, warm_calls=1)
    assert report['status'] == 'completed_with_failed_layouts'
    assert report['fastest_observed_layout'] == 0
    assert report['summaries'][0]['host_peak_rss_sum_max_bytes'] > 0
    assert not report['summaries'][1]['eligible']


def test_benchmark_vetoes_cross_chain_coupling(tmp_path):
    report = benchmark_hmc_chain_execution(
        HMCChainTarget('tests.test_hmc_chain_execution:coupled_target'), INITIAL, KERNEL,
        [HMCChainLayout(3, device='CPU')], output_dir=tmp_path / 'coupled', repetitions=1, warm_calls=1)
    assert report['fastest_observed_layout'] is None
    assert 'disagree with singleton' in report['cells'][0]['error']


def test_batched_kalman_reference_benchmark(tmp_path):
    report = benchmark_hmc_chain_execution(
        HMCChainTarget('bayesfilter.testing.hmc_chain_benchmark_targets:kalman_target'),
        [[.2, -1.05], [.21, -1.04]], replace(KERNEL, step_size=.03),
        [HMCChainLayout(2, device='CPU')], output_dir=tmp_path / 'kalman', repetitions=1, warm_calls=1)
    assert report['status'] == 'completed', report
    assert report['cells'][0]['maximum_value_score_error'] < 1e-10
