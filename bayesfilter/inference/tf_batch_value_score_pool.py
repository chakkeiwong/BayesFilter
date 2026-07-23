"""Persistent CPU workers for batch-native TensorFlow value/score targets."""

from __future__ import annotations

import concurrent.futures
import importlib
import multiprocessing
import os
import resource
import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping


_TARGET: Any | None = None
_READY_BARRIER: Any | None = None
_METADATA: Mapping[str, Any] | None = None


def _worker_environment(cores: int) -> Mapping[str, str]:
    threads = str(int(cores))
    return {
        "CUDA_VISIBLE_DEVICES": "-1",
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "OMP_NUM_THREADS": threads,
        "OPENBLAS_NUM_THREADS": threads,
        "MKL_NUM_THREADS": threads,
        "NUMEXPR_NUM_THREADS": threads,
        "TF_NUM_INTRAOP_THREADS": threads,
        "TF_NUM_INTEROP_THREADS": "1",
    }


def _worker_init(
    factory_path: str,
    factory_config: Mapping[str, Any],
    cores: int,
    batch_sizes: tuple[int, ...],
    barrier: Any,
) -> None:
    global _TARGET, _READY_BARRIER, _METADATA

    expected = _worker_environment(cores)
    mismatched = {
        key: os.environ.get(key) for key, value in expected.items() if os.environ.get(key) != value
    }
    if mismatched:
        raise RuntimeError("batch worker environment mismatch: " + ", ".join(sorted(mismatched)))
    # Thread counts are supplied through the environment before TensorFlow is
    # imported; changing them after import can initialize the context too
    # early and raises in spawned workers.
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError("batch-native CPU worker found a visible GPU")
    module_name, symbol_name = str(factory_path).split(":", 1)
    factory = getattr(importlib.import_module(module_name), symbol_name)
    target = factory(dict(factory_config))
    if getattr(target, "evaluation_policy", None) != "batch_native_tensorflow_no_row_mapping_v1":
        raise RuntimeError("worker target is not the reviewed batch-native route")
    _TARGET = target
    _READY_BARRIER = barrier
    _METADATA = {
        "pid": os.getpid(),
        "worker_backend": "batch_native_value_score",
        "evaluation_policy": target.evaluation_policy,
        "compiled_batch_sizes": list(batch_sizes),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tensorflow_gpu_devices": [],
        "target_signature": target.target_signature(),
        "adapter_signature": target.adapter_signature(),
    }


def _worker_ready() -> Mapping[str, Any]:
    if _TARGET is None or _READY_BARRIER is None or _METADATA is None:
        raise RuntimeError("batch-native worker initialization is incomplete")
    try:
        _READY_BARRIER.wait()
    except threading.BrokenBarrierError as exc:
        raise RuntimeError("batch-native worker startup barrier failed") from exc
    return {
        "metadata": dict(_METADATA),
        "ru_maxrss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
    }


def _worker_evaluate(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if _TARGET is None or _METADATA is None:
        raise RuntimeError("batch-native worker is unavailable")
    import tensorflow as tf

    rows = tf.io.parse_tensor(payload["rows"], out_type=tf.float64)
    rows = tf.ensure_shape(rows, [int(payload["row_count"]), int(_TARGET.parameter_dim)])
    started = time.perf_counter()
    value, score = _TARGET.batch_value_and_score(rows)
    tf.debugging.assert_all_finite(value, "batch-native worker value")
    tf.debugging.assert_all_finite(score, "batch-native worker score")
    return {
        "worker_index": int(payload["worker_index"]),
        "item_start": int(payload["item_start"]),
        "item_stop": int(payload["item_stop"]),
        "request_id": str(payload["request_id"]),
        "value": bytes(tf.io.serialize_tensor(value).numpy()),
        "score": bytes(tf.io.serialize_tensor(score).numpy()),
        "runtime_seconds": time.perf_counter() - started,
        "ru_maxrss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "metadata": dict(_METADATA),
    }


@dataclass(frozen=True)
class TFBatchValueScorePoolConfig:
    factory_path: str
    factory_config: Mapping[str, Any]
    dimension: int
    worker_count: int = 8
    cores_per_worker: int = 1
    batch_sizes: tuple[int, ...] = (1, 2, 8, 12, 13, 32)
    timeout_seconds: float = 900.0

    def __post_init__(self) -> None:
        if ":" not in self.factory_path:
            raise ValueError("factory_path must be module:callable")
        if int(self.dimension) <= 0 or int(self.worker_count) <= 0:
            raise ValueError("dimension and worker_count must be positive")
        if int(self.cores_per_worker) <= 0 or float(self.timeout_seconds) <= 0.0:
            raise ValueError("cores_per_worker and timeout_seconds must be positive")
        if any(int(size) <= 0 for size in self.batch_sizes):
            raise ValueError("batch_sizes must be positive")


class TFBatchValueScorePool:
    """Process-sharded rank-2 TensorFlow target evaluator."""

    def __init__(self, config: TFBatchValueScorePoolConfig) -> None:
        self.config = config
        self._executor: concurrent.futures.ProcessPoolExecutor | None = None
        self._startup: Mapping[str, Any] | None = None

    def __enter__(self) -> "TFBatchValueScorePool":
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def abort(self) -> None:
        executor = self._executor
        self._executor = None
        if executor is None:
            return
        processes = getattr(executor, "_processes", {})
        for process in tuple(processes.values()):
            if process.is_alive():
                process.terminate()
        executor.shutdown(wait=True, cancel_futures=True)

    def evaluate(
        self, rows: Any, *, request_id: str
    ) -> tuple[Any, Any, Mapping[str, Any]]:
        import tensorflow as tf

        matrix = tf.convert_to_tensor(rows, tf.float64)
        if matrix.shape.rank != 2 or matrix.shape[1] != int(self.config.dimension):
            raise ValueError("rows must have static shape [batch, dimension]")
        row_count = matrix.shape[0]
        if row_count is None or int(row_count) < int(self.config.worker_count):
            raise ValueError("batch must contain at least one row per configured worker")
        tf.debugging.assert_all_finite(matrix, "batch-native pool input")
        request = str(request_id)
        if not request:
            raise ValueError("request_id must be nonempty")
        self._ensure_started()
        assert self._executor is not None
        futures = []
        for worker_index in range(int(self.config.worker_count)):
            start = int(row_count) * worker_index // int(self.config.worker_count)
            stop = int(row_count) * (worker_index + 1) // int(self.config.worker_count)
            shard_size = stop - start
            if shard_size not in set(int(size) for size in self.config.batch_sizes):
                raise ValueError(f"undeclared batch-native shard size {shard_size}")
            serialized = bytes(tf.io.serialize_tensor(matrix[start:stop]).numpy())
            futures.append(
                self._executor.submit(
                    _worker_evaluate,
                    {
                        "worker_index": worker_index,
                        "item_start": start,
                        "item_stop": stop,
                        "row_count": shard_size,
                        "request_id": request,
                        "rows": serialized,
                    },
                )
            )
        try:
            results = [future.result(timeout=self.config.timeout_seconds) for future in futures]
        except BaseException:
            self.abort()
            raise
        results.sort(key=lambda row: int(row["item_start"]))
        values = tf.concat(
            [tf.io.parse_tensor(row["value"], out_type=tf.float64) for row in results], axis=0
        )
        scores = tf.concat(
            [tf.io.parse_tensor(row["score"], out_type=tf.float64) for row in results], axis=0
        )
        values = tf.ensure_shape(values, [int(row_count)])
        scores = tf.ensure_shape(scores, [int(row_count), int(self.config.dimension)])
        tf.debugging.assert_all_finite(values, "batch-native pooled values")
        tf.debugging.assert_all_finite(scores, "batch-native pooled scores")
        return values, scores, self._metadata(results, request=request)

    def _ensure_started(self) -> None:
        if self._executor is not None:
            return
        environment = _worker_environment(self.config.cores_per_worker)
        previous = {key: os.environ.get(key) for key in environment}
        os.environ.update(environment)
        try:
            context = multiprocessing.get_context("spawn")
            barrier = context.Barrier(
                int(self.config.worker_count), timeout=float(self.config.timeout_seconds)
            )
            self._executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=int(self.config.worker_count),
                mp_context=context,
                initializer=_worker_init,
                initargs=(
                    self.config.factory_path,
                    dict(self.config.factory_config),
                    int(self.config.cores_per_worker),
                    tuple(int(size) for size in self.config.batch_sizes),
                    barrier,
                ),
            )
            readiness = [
                self._executor.submit(_worker_ready)
                for _ in range(int(self.config.worker_count))
            ]
            rows = [future.result(timeout=self.config.timeout_seconds) for future in readiness]
        except BaseException:
            self.abort()
            raise
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        metadata = [dict(row["metadata"]) for row in rows]
        pids = sorted({int(row["pid"]) for row in metadata})
        if len(pids) != int(self.config.worker_count):
            self.abort()
            raise RuntimeError("not every batch-native worker initialized")
        if any(row["worker_backend"] != "batch_native_value_score" for row in metadata):
            self.abort()
            raise RuntimeError("worker backend admission failed")
        self._startup = {
            "startup_worker_pids": pids,
            "startup_worker_metadata": metadata,
            "startup_worker_ru_maxrss_sum_bytes": sum(
                int(row["ru_maxrss_bytes"]) for row in rows
            ),
        }

    def _metadata(
        self, results: list[Mapping[str, Any]], *, request: str
    ) -> Mapping[str, Any]:
        assert self._startup is not None
        return {
            "backend": "persistent_cpu_batch_native_tensorflow_pool",
            "evaluation_mode": "batch_native",
            "worker_backend": "batch_native_value_score",
            "configured_worker_count": int(self.config.worker_count),
            "compiled_batch_sizes": list(self.config.batch_sizes),
            "request_id": request,
            "worker_runtime_seconds": [float(row["runtime_seconds"]) for row in results],
            "worker_runtime_max_seconds": max(float(row["runtime_seconds"]) for row in results),
            "active_worker_ru_maxrss_sum_bytes": sum(
                int(row["ru_maxrss_bytes"]) for row in results
            ),
            **self._startup,
        }


__all__ = ["TFBatchValueScorePool", "TFBatchValueScorePoolConfig"]
