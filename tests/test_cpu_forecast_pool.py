from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np

from bayesfilter.inference.cpu_forecast_pool import (
    CPUForecastPool,
    CPUForecastPoolConfig,
)
from bayesfilter.nonlinear.ssl_lstm_complexity_predictive_tf import (
    complexity_forecast_worker_factory,
)


def _config(worker_count: int = 2) -> CPUForecastPoolConfig:
    return CPUForecastPoolConfig(
        worker_factory_path=(
            "bayesfilter.nonlinear.ssl_lstm_complexity_predictive_tf:"
            "complexity_forecast_worker_factory"
        ),
        worker_config={"q": 1},
        worker_count=worker_count,
        timeout_seconds=300.0,
    )


def test_forecast_pool_matches_native_scalar_order_and_replay() -> None:
    rows = np.asarray(
        [
            [0.35, -0.08, 0.65, 0.05],
            [0.36, -0.07, 0.64, 0.04],
        ],
        dtype=np.float64,
    )
    seeds = np.asarray(((20260719, 70001), (20260719, 70002)), dtype=np.int32)
    native = complexity_forecast_worker_factory({"q": 1})
    expected = [native.evaluate(row, seed) for row, seed in zip(rows, seeds, strict=True)]
    with CPUForecastPool(_config()) as pool:
        first = pool.evaluate(rows, seeds, request_id="forecast-parity")
        second = pool.evaluate(rows, seeds, request_id="forecast-replay")
    for index in range(3):
        expected_array = np.asarray([row[index].numpy() for row in expected])
        np.testing.assert_allclose(first[index], expected_array, rtol=0.0, atol=0.0)
        np.testing.assert_array_equal(second[index], first[index])
    metadata = first[3]
    assert metadata["configured_worker_count"] == 2
    assert len(metadata["startup_worker_pids"]) == 2
    assert metadata["active_worker_count"] == 2
    assert all(
        row["cuda_visible_devices"] == "-1"
        for row in metadata["worker_metadata"]
    )
    assert all(
        row["tensorflow_gpu_devices"] == []
        for row in metadata["worker_metadata"]
    )
    assert metadata["aggregate_parent_worker_ru_maxrss_bytes"] == (
        metadata["parent_ru_maxrss_bytes"]
        + metadata["worker_ru_maxrss_sum_bytes"]
    )


def test_forecast_pool_startup_barrier_initializes_full_pool_for_small_batch() -> None:
    rows = np.asarray([[0.35, -0.08, 0.65, 0.05]], dtype=np.float64)
    seeds = np.asarray(((20260719, 70101),), dtype=np.int32)
    with CPUForecastPool(_config(worker_count=3)) as pool:
        means, variances, observations, metadata = pool.evaluate(
            rows, seeds, request_id="small-batch"
        )
    assert means.shape == variances.shape == observations.shape == (1, 2, 10)
    assert metadata["active_worker_count"] == 1
    assert len(metadata["startup_worker_pids"]) == 3
    assert len(metadata["startup_worker_ru_maxrss_bytes_by_pid"]) == 3
