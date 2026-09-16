"""Persistent spawned CPU pool for q-general SSL-LSTM forecasts."""

from __future__ import annotations

import concurrent.futures
import hashlib
import importlib
import multiprocessing
import os
import resource
import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


_WORKER_FORECAST: Any | None = None
_WORKER_METADATA: Mapping[str, Any] | None = None
_WORKER_BARRIER: Any | None = None
_CPU_FORECAST_WORKER_ENV = "BAYESFILTER_CPU_FORECAST_WORKER"


def _worker_environment(cores: int) -> dict[str, str]:
    threads = str(max(1, int(cores)))
    return {
        "CUDA_VISIBLE_DEVICES": "-1",
        _CPU_FORECAST_WORKER_ENV: "1",
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "OMP_NUM_THREADS": threads,
        "OPENBLAS_NUM_THREADS": threads,
        "MKL_NUM_THREADS": threads,
        "NUMEXPR_NUM_THREADS": threads,
        "TF_NUM_INTRAOP_THREADS": threads,
        "TF_NUM_INTEROP_THREADS": "1",
    }


def _array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _worker_init(
    factory_path: str,
    factory_config: Mapping[str, Any],
    cores: int,
    startup_barrier: Any,
) -> None:
    expected = _worker_environment(cores)
    mismatched = {
        key: os.environ.get(key)
        for key, value in expected.items()
        if os.environ.get(key) != value
    }
    if mismatched:
        raise RuntimeError(
            "CPU forecast worker environment mismatch: "
            + ", ".join(sorted(mismatched))
        )
    import tensorflow as tf

    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        pass
    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError("CPU forecast worker found a visible GPU")
    module_name, symbol_name = str(factory_path).split(":", 1)
    factory = getattr(importlib.import_module(module_name), symbol_name)
    worker = factory(dict(factory_config))
    method = getattr(worker, "evaluate", None)
    if not callable(method):
        raise RuntimeError("forecast worker factory must expose evaluate")
    global _WORKER_FORECAST, _WORKER_METADATA, _WORKER_BARRIER
    _WORKER_FORECAST = method
    _WORKER_METADATA = {
        "pid": int(os.getpid()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tensorflow_gpu_devices": [],
        "jit_compile": True,
        "backend": "scalar_q_general_principal_root_forecast_xla",
    }
    _WORKER_BARRIER = startup_barrier


def _worker_ready() -> Mapping[str, Any]:
    if _WORKER_METADATA is None or _WORKER_BARRIER is None:
        raise RuntimeError("CPU forecast worker is not initialized")
    try:
        _WORKER_BARRIER.wait()
    except threading.BrokenBarrierError as exc:
        raise RuntimeError("CPU forecast startup barrier failed") from exc
    return {
        "worker_metadata": dict(_WORKER_METADATA),
        "ru_maxrss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
    }


def _worker_eval(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if _WORKER_FORECAST is None or _WORKER_METADATA is None:
        raise RuntimeError("CPU forecast worker is not initialized")
    import tensorflow as tf

    rows = np.asarray(payload["rows"], dtype=np.float64)
    seeds = np.asarray(payload["seeds"], dtype=np.int32)
    started = time.perf_counter()
    means = []
    variances = []
    observations = []
    for row, seed in zip(rows, seeds, strict=True):
        mean, variance, observation = _WORKER_FORECAST(
            tf.convert_to_tensor(row, tf.float64),
            tf.convert_to_tensor(seed, tf.int32),
        )
        means.append(np.asarray(mean.numpy(), dtype=np.float64))
        variances.append(np.asarray(variance.numpy(), dtype=np.float64))
        observations.append(np.asarray(observation.numpy(), dtype=np.float64))
    return {
        "worker_index": int(payload["worker_index"]),
        "item_start": int(payload["item_start"]),
        "item_stop": int(payload["item_stop"]),
        "request_id": str(payload["request_id"]),
        "rows_hash": str(payload["rows_hash"]),
        "seeds_hash": str(payload["seeds_hash"]),
        "shard_rows_hash": _array_hash(rows),
        "shard_seeds_hash": _array_hash(seeds),
        "conditional_means": np.asarray(means, dtype=np.float64),
        "conditional_variances": np.asarray(variances, dtype=np.float64),
        "observations": np.asarray(observations, dtype=np.float64),
        "runtime_seconds": float(time.perf_counter() - started),
        "ru_maxrss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "worker_metadata": dict(_WORKER_METADATA),
    }


@dataclass(frozen=True)
class CPUForecastPoolConfig:
    worker_factory_path: str
    worker_config: Mapping[str, Any]
    worker_count: int
    cores_per_worker: int = 1
    timeout_seconds: float = 7200.0

    def __post_init__(self) -> None:
        if ":" not in str(self.worker_factory_path):
            raise ValueError("worker_factory_path must be module:callable")
        if int(self.worker_count) <= 0 or int(self.cores_per_worker) <= 0:
            raise ValueError("worker and core counts must be positive")
        if float(self.timeout_seconds) <= 0.0:
            raise ValueError("timeout_seconds must be positive")


class CPUForecastPool:
    """Persistent ordered scalar forecast pool."""

    def __init__(self, config: CPUForecastPoolConfig) -> None:
        self.config = config
        self._executor: concurrent.futures.ProcessPoolExecutor | None = None
        self._barrier: Any | None = None
        self._startup_metadata: Mapping[str, Any] | None = None
        self._opened = False

    def __enter__(self) -> "CPUForecastPool":
        if self._opened:
            raise RuntimeError("CPU forecast pool is already open")
        self._opened = True
        return self

    def evaluate(
        self,
        rows: Any,
        seeds: Any,
        *,
        request_id: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, Mapping[str, Any]]:
        if not self._opened:
            raise RuntimeError("CPU forecast pool must be opened as a context manager")
        matrix = np.asarray(rows, dtype=np.float64)
        seed_matrix = np.asarray(seeds, dtype=np.int32)
        if matrix.ndim != 2 or matrix.shape[1] != 4:
            raise ValueError("forecast rows must have shape [batch,4]")
        if seed_matrix.shape != (matrix.shape[0], 2):
            raise ValueError("forecast seeds must have shape [batch,2]")
        if matrix.shape[0] == 0 or not np.all(np.isfinite(matrix)):
            raise ValueError("forecast rows must be nonempty and finite")
        request = str(request_id)
        if not request:
            raise ValueError("request_id must be nonempty")
        rows_hash = _array_hash(matrix)
        seeds_hash = _array_hash(seed_matrix)
        worker_count = min(int(self.config.worker_count), int(matrix.shape[0]))
        environment = _worker_environment(self.config.cores_per_worker)
        previous = {key: os.environ.get(key) for key in environment}
        os.environ.update(environment)
        try:
            if self._executor is None:
                context = multiprocessing.get_context("spawn")
                self._barrier = context.Barrier(
                    int(self.config.worker_count),
                    timeout=float(self.config.timeout_seconds),
                )
                self._executor = concurrent.futures.ProcessPoolExecutor(
                    max_workers=int(self.config.worker_count),
                    mp_context=context,
                    initializer=_worker_init,
                    initargs=(
                        self.config.worker_factory_path,
                        dict(self.config.worker_config),
                        int(self.config.cores_per_worker),
                        self._barrier,
                    ),
                )
                readiness = [
                    self._executor.submit(_worker_ready)
                    for _ in range(int(self.config.worker_count))
                ]
                ready = [
                    future.result(timeout=float(self.config.timeout_seconds))
                    for future in readiness
                ]
                startup_rss_by_pid = {
                    int(row["worker_metadata"]["pid"]): int(row["ru_maxrss_bytes"])
                    for row in ready
                }
                pids = sorted(startup_rss_by_pid)
                if len(pids) != int(self.config.worker_count):
                    raise RuntimeError("CPU forecast pool did not initialize every worker")
                self._startup_metadata = {
                    "configured_worker_count": int(self.config.worker_count),
                    "startup_worker_pids": pids,
                    "startup_worker_ru_maxrss_bytes_by_pid": {
                        str(pid): startup_rss_by_pid[pid] for pid in pids
                    },
                    "startup_worker_ru_maxrss_sum_bytes": int(
                        sum(startup_rss_by_pid.values())
                    ),
                }
            futures = []
            for worker_index in range(worker_count):
                start = (matrix.shape[0] * worker_index) // worker_count
                stop = (matrix.shape[0] * (worker_index + 1)) // worker_count
                futures.append(
                    self._executor.submit(
                        _worker_eval,
                        {
                            "worker_index": worker_index,
                            "item_start": start,
                            "item_stop": stop,
                            "request_id": request,
                            "rows_hash": rows_hash,
                            "seeds_hash": seeds_hash,
                            "rows": matrix[start:stop],
                            "seeds": seed_matrix[start:stop],
                        },
                    )
                )
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        results = [
            future.result(timeout=float(self.config.timeout_seconds))
            for future in futures
        ]
        results.sort(key=lambda row: int(row["item_start"]))
        expected_start = 0
        for row in results:
            start = int(row["item_start"])
            stop = int(row["item_stop"])
            if start != expected_start or stop <= start:
                raise RuntimeError("forecast worker shard coverage mismatch")
            if row["request_id"] != request or row["rows_hash"] != rows_hash:
                raise RuntimeError("forecast worker request identity mismatch")
            if row["seeds_hash"] != seeds_hash:
                raise RuntimeError("forecast worker seed identity mismatch")
            if row["shard_rows_hash"] != _array_hash(matrix[start:stop]):
                raise RuntimeError("forecast worker row content mismatch")
            if row["shard_seeds_hash"] != _array_hash(seed_matrix[start:stop]):
                raise RuntimeError("forecast worker seed content mismatch")
            expected_start = stop
        if expected_start != matrix.shape[0]:
            raise RuntimeError("forecast worker coverage is incomplete")
        means = np.concatenate([row["conditional_means"] for row in results], axis=0)
        variances = np.concatenate(
            [row["conditional_variances"] for row in results], axis=0
        )
        observations = np.concatenate([row["observations"] for row in results], axis=0)
        expected_shape = (matrix.shape[0], 2, 10)
        if means.shape != expected_shape or variances.shape != expected_shape:
            raise RuntimeError("forecast worker conditional-moment shape mismatch")
        if observations.shape != expected_shape:
            raise RuntimeError("forecast worker observation shape mismatch")
        if not all(np.all(np.isfinite(row)) for row in (means, variances, observations)):
            raise FloatingPointError("forecast worker returned nonfinite output")
        if not np.all(variances > 0.0):
            raise FloatingPointError("forecast worker returned nonpositive variance")
        if self._startup_metadata is None:
            raise RuntimeError("forecast startup metadata is unavailable")
        startup_rss_by_pid = {
            int(pid): int(value)
            for pid, value in self._startup_metadata[
                "startup_worker_ru_maxrss_bytes_by_pid"
            ].items()
        }
        active_rss_by_pid = {
            int(row["worker_metadata"]["pid"]): int(row["ru_maxrss_bytes"])
            for row in results
        }
        worker_ru_maxrss_sum = sum(
            max(startup_rss_by_pid[pid], active_rss_by_pid.get(pid, 0))
            for pid in startup_rss_by_pid
        )
        parent_ru_maxrss = int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
        metadata = {
            "request_id": request,
            "rows_hash": rows_hash,
            "seeds_hash": seeds_hash,
            "active_worker_count": len(results),
            "worker_pids": sorted(
                {int(row["worker_metadata"]["pid"]) for row in results}
            ),
            "worker_metadata": [dict(row["worker_metadata"]) for row in results],
            "worker_runtime_seconds": [float(row["runtime_seconds"]) for row in results],
            "worker_runtime_max_seconds": max(
                float(row["runtime_seconds"]) for row in results
            ),
            "active_worker_ru_maxrss_sum_bytes": int(
                sum(int(row["ru_maxrss_bytes"]) for row in results)
            ),
            "worker_ru_maxrss_sum_bytes": int(worker_ru_maxrss_sum),
            "parent_ru_maxrss_bytes": parent_ru_maxrss,
            "aggregate_parent_worker_ru_maxrss_bytes": int(
                parent_ru_maxrss + worker_ru_maxrss_sum
            ),
            "backend": "persistent_cpu_principal_root_forecast_pool",
            **dict(self._startup_metadata),
        }
        return means, variances, observations, metadata

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        self._barrier = None
        self._startup_metadata = None
        self._opened = False

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.close()


__all__ = ["CPUForecastPool", "CPUForecastPoolConfig"]
