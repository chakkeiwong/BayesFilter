"""Persistent compiled batched workers: identity, affinity, failures and RSS."""
import concurrent.futures
import os
import signal

import pytest
import tensorflow as tf

from bayesfilter.inference.tf_batch_value_score_pool import TFBatchValueScorePool, TFBatchValueScorePoolConfig


class QuadraticTarget:
    evaluation_policy = 'batch_native_tensorflow_status_no_row_mapping_v2'
    parameter_dim = 2
    _jit_compile = True

    def target_signature(self):
        return '1'*64

    def adapter_signature(self):
        return '2'*64

    def neutra_batch_log_prob_and_grad_status(self, rows):
        valid = rows[:, 0] >= 0
        value = -.5*tf.reduce_sum(rows**2, axis=1)
        return tf.where(valid, value, tf.constant(float('nan'), tf.float64)), -rows, {'eligible': valid}


def quadratic_factory(_config):
    return QuadraticTarget()


def config():
    cpus = tuple(sorted(os.sched_getaffinity(0)))
    if len(cpus) < 4:
        pytest.skip('four CPUs needed for this worker-affinity fixture')
    return TFBatchValueScorePoolConfig(factory_path='tests.test_tf_batch_value_score_pool_groups:quadratic_factory',
        factory_config={}, dimension=2, worker_count=2, cores_per_worker=2,
        worker_cpu_groups=(cpus[:2], cpus[2:4]), batch_sizes=(8,), batch_per_worker=8, timeout_seconds=60)


def test_persistent_core_groups_ordered_rows_concurrency_and_memory():
    rows = tf.reshape(tf.cast(tf.range(48), tf.float64), [24, 2])/100
    submitted, received = [], []
    with TFBatchValueScorePool(config()) as pool:
        value, score, status, first = pool.evaluate_with_status(rows, request_id='first',
            on_submit=lambda start, stop: submitted.append((start, stop)), on_result=received.append)
        tf.debugging.assert_near(value, -.5*tf.reduce_sum(rows**2, axis=1), atol=1e-12, rtol=1e-12)
        tf.debugging.assert_equal(score, -rows)
        assert bool(tf.reduce_all(status['eligible']))
        assert submitted == [(0, 8), (8, 16), (16, 24)]
        assert sorted((r['item_start'], r['item_stop']) for r in received) == submitted
        expected_rss = sum(max(r['ru_maxrss_bytes'] for r in received if r['worker_pid'] == pid)
                           for pid in {r['worker_pid'] for r in received})
        assert first['active_worker_ru_maxrss_sum_bytes'] == expected_rss
        assert all(counts == {'8': 1} for counts in first['worker_status_trace_counts'])
        for worker in first['startup_worker_metadata']:
            assert worker['status_jit_compile']
            assert {tuple(t['affinity']) for t in worker['thread_affinity']} == {tuple(worker['assigned_cpus'])}
        with concurrent.futures.ThreadPoolExecutor(2) as clients:
            futures = [clients.submit(pool.evaluate_with_status, rows+offset, request_id=f'client-{offset}')
                       for offset in (1., 2.)]
            for offset, future in zip((1., 2.), futures, strict=True):
                _value, next_score, _status, metadata = future.result()
                tf.debugging.assert_equal(next_score, -(rows+offset))
                assert metadata['startup_worker_pids'] == first['startup_worker_pids']
                assert all(counts == {'8': 1} for counts in metadata['worker_status_trace_counts'])
        _value, _score, invalid, _metadata = pool.evaluate_with_status(-rows-1, request_id='invalid')
        assert not bool(tf.reduce_any(invalid['eligible']))
        # A completed invalid target is retained data; it does not kill the pool.
        pool.evaluate_with_status(rows, request_id='after-invalid')


def test_worker_loss_aborts_pool_and_cannot_trigger_automatic_respawn():
    with TFBatchValueScorePool(config()) as pool:
        _, _, _, metadata = pool.evaluate_with_status(tf.ones([16, 2], tf.float64), request_id='before-loss')
        os.kill(metadata['startup_worker_pids'][0], signal.SIGTERM)
        with pytest.raises(Exception):
            pool.evaluate_with_status(tf.ones([16, 2], tf.float64), request_id='after-loss')
        assert pool._aborted
        with pytest.raises(RuntimeError, match='cannot restart'):
            pool.evaluate_with_status(tf.ones([16, 2], tf.float64), request_id='must-not-retry')


def test_response_from_another_request_is_rejected(monkeypatch):
    class WrongExecutor:
        _processes = {}

        def submit(self, function, payload):
            future = concurrent.futures.Future()
            future.set_result({**payload, 'request_id': 'stale-request'})
            return future

        def shutdown(self, **kwargs):
            pass

    pool = TFBatchValueScorePool(config())
    pool._executor = WrongExecutor()
    monkeypatch.setattr(pool, '_ensure_started', lambda: None)
    with pytest.raises(RuntimeError, match='identity/order'):
        pool.evaluate_with_status(tf.ones([16, 2], tf.float64), request_id='current-request')
    assert pool._aborted


def test_submission_failure_preserves_already_completed_results(monkeypatch):
    class PartialExecutor:
        _processes = {}

        def submit(self, function, payload):
            if payload['item_start']:
                raise RuntimeError('submission failed')
            future = concurrent.futures.Future()
            future.set_result(payload)
            return future

        def shutdown(self, **kwargs):
            pass

    pool = TFBatchValueScorePool(config())
    pool._executor = PartialExecutor()
    monkeypatch.setattr(pool, '_ensure_started', lambda: None)
    received, reserved = [], []
    with pytest.raises(RuntimeError, match='submission failed'):
        pool.evaluate_with_status(tf.ones([16, 2], tf.float64), request_id='partial',
            on_submit=lambda start, stop: reserved.append(start), on_result=received.append)
    assert pool._aborted
    assert reserved == [0, 8]
    assert [row['item_start'] for row in received] == [0]


def test_short_requests_rotate_across_workers_without_changing_row_order(monkeypatch):
    assigned = []
    class Executor:
        def __init__(self, index): self.index = index
        def submit(self, function, payload):
            assigned.append(self.index)
            rows = tf.io.parse_tensor(payload['rows'], out_type=tf.float64)
            future = concurrent.futures.Future()
            future.set_result({**payload, 'value': tf.io.serialize_tensor(tf.reduce_sum(rows, -1)).numpy(),
                'score': tf.io.serialize_tensor(-rows).numpy()})
            return future
    pool = TFBatchValueScorePool(config())
    pool._pinned_executors = (Executor(0), Executor(1))
    monkeypatch.setattr(pool, '_ensure_started', lambda: None)
    monkeypatch.setattr(pool, '_metadata', lambda *args, **kwargs: {})
    rows = tf.reshape(tf.cast(tf.range(48), tf.float64), [24, 2])
    for i in range(2):
        _value, score, _metadata = pool.evaluate(rows+i, request_id=str(i))
        tf.debugging.assert_equal(score, -(rows+i))
    assert assigned == [0, 1, 0, 1, 0, 1]
