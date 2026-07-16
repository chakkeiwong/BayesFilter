from __future__ import annotations

import os

import numpy as np
import pytest

from bayesfilter.inference.cpu_xla_cloud import (
    CPUXLACloudConfig,
    CPUXLACloudEvaluator,
    default_cpu_worker_count,
)


FACTORY = (
    "bayesfilter.testing.fixed_center_cpu_xla_fixture:"
    "quadratic_value_score_factory"
)


def test_default_worker_count_uses_one_third_and_allows_override() -> None:
    assert default_cpu_worker_count(task_count=20, detected_cores=30) == 10
    assert default_cpu_worker_count(task_count=4, detected_cores=30) == 4
    assert (
        default_cpu_worker_count(
            task_count=20, detected_cores=30, worker_override=17
        )
        == 17
    )
    with pytest.raises(ValueError, match="worker_override"):
        default_cpu_worker_count(
            task_count=4, detected_cores=30, worker_override=0
        )


def test_spawned_cpu_xla_pool_is_persistent_ordered_and_explicit() -> None:
    precision = np.array([[2.0, 0.5], [0.5, 3.0]])
    center = np.array([0.2, -0.1])
    points = np.array(
        [[0.0, 0.0], [0.1, -0.2], [0.4, 0.3], [-0.2, 0.1]], dtype=float
    )
    config = CPUXLACloudConfig(
        worker_factory_path=FACTORY,
        dimension=2,
        worker_count=2,
        set_affinity=False,
        heartbeat_seconds=30.0,
        factory_config={"precision": precision.tolist(), "center": center.tolist()},
    )
    previous_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = "parent-marker"
    try:
        with CPUXLACloudEvaluator(config) as evaluator:
            events = []
            first = evaluator.evaluate(points, progress_callback=events.append)
            assert os.environ["CUDA_VISIBLE_DEVICES"] == "parent-marker"
            second = evaluator.evaluate(points[::-1])
    finally:
        if previous_cuda is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous_cuda

    delta = points - center[None, :]
    expected_scores = -delta @ precision.T
    expected_values = 0.5 * np.sum(delta * expected_scores, axis=1)
    np.testing.assert_allclose(first.values, expected_values)
    np.testing.assert_allclose(first.scores, expected_scores)
    np.testing.assert_allclose(second.values, expected_values[::-1])
    np.testing.assert_allclose(second.scores, expected_scores[::-1])
    assert first.worker_count == 2
    assert first.batch_size == 1
    assert first.jit_compile is True
    assert first.automatic_fallback_used is False
    assert first.worker_pids
    assert second.worker_pids
    assert set(second.worker_pids).issubset(set(first.worker_pids) | set(second.worker_pids))
    assert first.payload()["cuda_visible_devices"] == "-1"
    assert first.payload()["automatic_fallback_used"] is False
    assert first.worker_bootstrap_records
    assert all(
        record["cpu_env_inherited_before_initializer"] is True
        and record["framework_modules_at_initializer_entry"] == []
        and record["cuda_visible_devices"] == "-1"
        and record["tf_force_gpu_allow_growth"] == "true"
        and record["jit_compile"] is True
        and record["batch_size"] == 1
        for record in first.worker_bootstrap_records
    )
    assert events[0]["stage"] == "cpu_xla_cloud_evaluation_started"
    row_events = [
        event for event in events if event["stage"] == "cpu_xla_cloud_row_completed"
    ]
    assert [event["completed_rows"] for event in row_events] == [1, 2, 3, 4]
    assert all(event["semantic_progress"] is True for event in row_events)
    assert events[-1]["stage"] == "cpu_xla_cloud_evaluation_completed"


def test_pool_lifecycle_and_child_initialization_fail_closed() -> None:
    evaluator = CPUXLACloudEvaluator(
        CPUXLACloudConfig(
            worker_factory_path="missing_module:missing_factory",
            dimension=2,
            worker_count=1,
            set_affinity=False,
        )
    )
    with evaluator:
        with pytest.raises(Exception):
            evaluator.evaluate(np.zeros((1, 2)))
    with pytest.raises(RuntimeError, match="context manager"):
        evaluator.evaluate(np.zeros((1, 2)))


def test_worker_bootstrap_rejects_missing_inherited_cpu_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bayesfilter.cpu_xla_worker_bootstrap import initialize_cpu_xla_worker

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    with pytest.raises(RuntimeError, match="must inherit"):
        initialize_cpu_xla_worker(FACTORY, 2, {}, False)


def test_default_pool_clamps_to_task_count_and_heartbeat_is_not_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bayesfilter.inference.cpu_xla_cloud as cloud

    real_wait = cloud.concurrent.futures.wait
    wait_calls = 0

    def one_heartbeat_then_wait(fs, **kwargs):
        nonlocal wait_calls
        wait_calls += 1
        if wait_calls == 1:
            return set(), set(fs)
        return real_wait(fs, **kwargs)

    monkeypatch.setattr(cloud.concurrent.futures, "wait", one_heartbeat_then_wait)
    config = CPUXLACloudConfig(
        worker_factory_path=FACTORY,
        dimension=2,
        worker_count=None,
        set_affinity=False,
        heartbeat_seconds=30.0,
        factory_config={"precision": np.eye(2).tolist(), "center": [0.0, 0.0]},
    )
    events = []
    with CPUXLACloudEvaluator(config) as evaluator:
        result = evaluator.evaluate(np.array([[0.1, -0.2]]), progress_callback=events.append)

    assert result.worker_count == 1
    heartbeat = next(
        event
        for event in events
        if event["stage"] == "cpu_xla_cloud_liveness_heartbeat"
    )
    assert heartbeat["completed_rows"] == 0
    assert heartbeat["semantic_progress"] is False
    assert heartbeat["heartbeat_is_not_progress"] is True
