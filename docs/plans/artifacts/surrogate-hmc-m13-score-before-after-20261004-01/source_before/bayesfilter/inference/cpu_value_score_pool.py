"""Persistent CPU process pool for scalar value/score target evaluations.

The pool is the execution boundary used by the DSGE-HMC-style NeuTra training
route.  Spawned workers hide CUDA before importing TensorFlow, construct one
target, and evaluate scalar rows eagerly.  The parent process owns the
transport, optimizer, and custom-gradient bridge.
"""

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


_WORKER_TARGET: Any | None = None
_WORKER_VALUE: Any | None = None
_WORKER_METADATA: Mapping[str, Any] | None = None
_WORKER_STARTUP_BARRIER: Any | None = None
_WORKER_BATCH_TARGETS: dict[int, Any] = {}
_WORKER_BATCH_VALUES: dict[int, Any] = {}
_CPU_WORKER_ENV = "BAYESFILTER_CPU_VALUE_SCORE_WORKER"


def _cpu_worker_environment(cores: int) -> dict[str, str]:
    threads = str(max(1, int(cores)))
    return {
        "CUDA_VISIBLE_DEVICES": "-1",
        _CPU_WORKER_ENV: "1",
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "OMP_NUM_THREADS": threads,
        "OPENBLAS_NUM_THREADS": threads,
        "MKL_NUM_THREADS": threads,
        "NUMEXPR_NUM_THREADS": threads,
        "TF_NUM_INTRAOP_THREADS": threads,
        "TF_NUM_INTEROP_THREADS": "1",
    }


def _array_hash(rows: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(rows, dtype=np.float64)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _worker_init(
    factory_path: str,
    factory_config: Mapping[str, Any],
    cores: int,
    startup_barrier: Any,
    evaluation_mode: str,
    batch_sizes: tuple[int, ...],
) -> None:
    """Initialize a worker with fail-closed CPU visibility."""

    global _WORKER_TARGET, _WORKER_VALUE, _WORKER_BATCH_TARGETS, _WORKER_BATCH_VALUES
    global _WORKER_METADATA, _WORKER_STARTUP_BARRIER

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("CPU value/score worker must inherit CUDA_VISIBLE_DEVICES=-1")
    expected_environment = _cpu_worker_environment(cores)
    mismatched = {
        key: os.environ.get(key)
        for key, value in expected_environment.items()
        if os.environ.get(key) != value
    }
    if mismatched:
        raise RuntimeError(
            "CPU value/score worker did not inherit its thread environment: "
            + ", ".join(sorted(mismatched))
        )
    import tensorflow as tf

    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        pass
    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError("CPU value/score worker found a visible GPU")
    module_name, symbol_name = str(factory_path).split(":", 1)
    factory = getattr(importlib.import_module(module_name), symbol_name)
    target = factory(dict(factory_config))
    scalar_value_score = None
    scalar_value = None
    batch_value_scores: dict[int, Any] = {}
    batch_values: dict[int, Any] = {}
    if evaluation_mode == "batch_native":
        if not batch_sizes:
            raise ValueError("batch_native workers require static batch sizes")

        for size in batch_sizes:
            static_size = int(size)

            @tf.function(
                input_signature=(
                    tf.TensorSpec([static_size, int(target.parameter_dim)], tf.float64),
                ),
                jit_compile=False,
                reduce_retracing=False,
            )
            def batch_value_score_fn(rows: Any) -> tuple[Any, Any]:
                return target.batch_value_and_score(rows)

            @tf.function(
                input_signature=(
                    tf.TensorSpec([static_size, int(target.parameter_dim)], tf.float64),
                ),
                jit_compile=False,
                reduce_retracing=False,
            )
            def batch_value_fn(rows: Any) -> Any:
                values, _scores = target.batch_value_and_score(rows)
                return values

            sample = tf.zeros([static_size, int(target.parameter_dim)], tf.float64)
            batch_value_score_fn(sample)
            batch_value_fn(sample)
            batch_value_scores[static_size] = batch_value_score_fn
            batch_values[static_size] = batch_value_fn
    else:
        method = getattr(target, "eager_value_and_score", None)
        if not callable(method):
            method = target.value_and_score

        @tf.function(
            input_signature=(tf.TensorSpec([int(target.parameter_dim)], tf.float64),),
            jit_compile=False,
            reduce_retracing=True,
        )
        def scalar_value_score(row: Any) -> tuple[Any, Any]:
            return method(row)

        value_method = getattr(target, "eager_value", None)
        if not callable(value_method):
            value_method = lambda row: method(row)[0]

        @tf.function(
            input_signature=(tf.TensorSpec([int(target.parameter_dim)], tf.float64),),
            jit_compile=False,
            reduce_retracing=True,
        )
        def scalar_value(row: Any) -> Any:
            return value_method(row)

        target_config = getattr(target, "config", None)
        prior_center = getattr(target_config, "prior_center", None)
        if prior_center is None:
            raise RuntimeError("CPU value/score worker target requires a warmup point")
        warmup = tf.ensure_shape(
            tf.convert_to_tensor(prior_center, tf.float64),
            [int(target.parameter_dim)],
        )
        scalar_value_score(warmup)
        scalar_value(warmup)
    _WORKER_TARGET = scalar_value_score
    _WORKER_VALUE = scalar_value
    _WORKER_BATCH_TARGETS = batch_value_scores
    _WORKER_BATCH_VALUES = batch_values
    _WORKER_METADATA = {
        "pid": int(os.getpid()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "jit_compile": False,
        "tensorflow_gpu_devices": [],
        "worker_backend": (
            "batch_native_value_score"
            if evaluation_mode == "batch_native"
            else "scalar_eager_value_score"
        ),
        "evaluation_mode": str(evaluation_mode),
        "compiled_batch_sizes": [int(size) for size in batch_sizes],
        "target_signature": str(target.target_signature())
        if callable(getattr(target, "target_signature", None))
        else "",
        "adapter_signature": str(target.adapter_signature())
        if callable(getattr(target, "adapter_signature", None))
        else "",
        "evaluation_policy": str(getattr(target, "evaluation_policy", "")),
        "source_hashes": dict(
            getattr(target, "signature_payload", lambda: {})().get("source_hashes", {})
        ),
    }
    _WORKER_STARTUP_BARRIER = startup_barrier


def _worker_ready() -> Mapping[str, Any]:
    """Hold one task per process until the complete pool is initialized."""

    if _WORKER_METADATA is None or _WORKER_STARTUP_BARRIER is None:
        raise RuntimeError("CPU value/score worker was not initialized")
    try:
        _WORKER_STARTUP_BARRIER.wait()
    except threading.BrokenBarrierError as exc:
        raise RuntimeError(
            "CPU value/score worker startup barrier failed"
        ) from exc
    return {
        "worker_metadata": dict(_WORKER_METADATA),
        "ru_maxrss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
    }


def _worker_eval(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if _WORKER_TARGET is None or _WORKER_METADATA is None:
        raise RuntimeError("CPU value/score worker was not initialized")
    import tensorflow as tf

    rows = np.asarray(payload["rows"], dtype=np.float64)
    started = time.perf_counter()
    values: list[float] = []
    scores: list[np.ndarray] = []
    for row in rows:
        value, score = _WORKER_TARGET(tf.convert_to_tensor(row, tf.float64))
        value_np = float(tf.convert_to_tensor(value, tf.float64).numpy())
        score_np = np.asarray(tf.convert_to_tensor(score, tf.float64).numpy(), dtype=np.float64)
        values.append(value_np)
        scores.append(score_np)
    values_np = np.asarray(values, dtype=np.float64)
    scores_np = np.asarray(scores, dtype=np.float64)
    return {
        "worker_index": int(payload["worker_index"]),
        "item_start": int(payload["item_start"]),
        "item_stop": int(payload["item_stop"]),
        "request_id": str(payload["request_id"]),
        "input_hash": str(payload["input_hash"]),
        "shard_hash": _array_hash(rows),
        "values": values_np,
        "scores": scores_np,
        "runtime_seconds": float(time.perf_counter() - started),
        "ru_maxrss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
        "worker_metadata": dict(_WORKER_METADATA),
    }


def _worker_eval_value(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if _WORKER_VALUE is None or _WORKER_METADATA is None:
        raise RuntimeError("CPU value worker was not initialized")
    import tensorflow as tf

    rows = np.asarray(payload["rows"], dtype=np.float64)
    started = time.perf_counter()
    values = np.asarray(
        [
            float(
                tf.convert_to_tensor(
                    _WORKER_VALUE(tf.convert_to_tensor(row, tf.float64)),
                    tf.float64,
                ).numpy()
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    return {
        "worker_index": int(payload["worker_index"]),
        "item_start": int(payload["item_start"]),
        "item_stop": int(payload["item_stop"]),
        "request_id": str(payload["request_id"]),
        "input_hash": str(payload["input_hash"]),
        "shard_hash": _array_hash(rows),
        "values": values,
        "runtime_seconds": float(time.perf_counter() - started),
        "ru_maxrss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
        "worker_metadata": dict(_WORKER_METADATA),
    }


def _worker_eval_batch(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if _WORKER_METADATA is None:
        raise RuntimeError("batch-native CPU value/score worker was not initialized")
    import tensorflow as tf

    rows = np.asarray(payload["rows"], dtype=np.float64)
    started = time.perf_counter()
    batch_target = _WORKER_BATCH_TARGETS.get(int(rows.shape[0]))
    if batch_target is None:
        raise ValueError(f"undeclared batch-native worker shard size {rows.shape[0]}")
    values, scores = batch_target(tf.convert_to_tensor(rows, tf.float64))
    return {
        "worker_index": int(payload["worker_index"]),
        "item_start": int(payload["item_start"]),
        "item_stop": int(payload["item_stop"]),
        "request_id": str(payload["request_id"]),
        "input_hash": str(payload["input_hash"]),
        "shard_hash": _array_hash(rows),
        "values": np.asarray(values.numpy(), dtype=np.float64),
        "scores": np.asarray(scores.numpy(), dtype=np.float64),
        "runtime_seconds": float(time.perf_counter() - started),
        "ru_maxrss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "worker_metadata": dict(_WORKER_METADATA),
    }


def _worker_eval_value_batch(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if _WORKER_METADATA is None:
        raise RuntimeError("batch-native CPU value worker was not initialized")
    import tensorflow as tf

    rows = np.asarray(payload["rows"], dtype=np.float64)
    started = time.perf_counter()
    batch_value = _WORKER_BATCH_VALUES.get(int(rows.shape[0]))
    if batch_value is None:
        raise ValueError(f"undeclared batch-native worker shard size {rows.shape[0]}")
    values = batch_value(tf.convert_to_tensor(rows, tf.float64))
    return {
        "worker_index": int(payload["worker_index"]),
        "item_start": int(payload["item_start"]),
        "item_stop": int(payload["item_stop"]),
        "request_id": str(payload["request_id"]),
        "input_hash": str(payload["input_hash"]),
        "shard_hash": _array_hash(rows),
        "values": np.asarray(values.numpy(), dtype=np.float64),
        "runtime_seconds": float(time.perf_counter() - started),
        "ru_maxrss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "worker_metadata": dict(_WORKER_METADATA),
    }


@dataclass(frozen=True)
class CPUValueScorePoolConfig:
    worker_factory_path: str
    worker_config: Mapping[str, Any]
    dimension: int
    worker_count: int = 4
    cores_per_worker: int = 1
    timeout_seconds: float = 600.0
    evaluation_mode: str = "scalar"
    batch_sizes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if ":" not in str(self.worker_factory_path):
            raise ValueError("worker_factory_path must be module:callable")
        if int(self.dimension) <= 0 or int(self.worker_count) <= 0:
            raise ValueError("dimension and worker_count must be positive")
        if int(self.cores_per_worker) <= 0 or float(self.timeout_seconds) <= 0.0:
            raise ValueError("cores_per_worker and timeout_seconds must be positive")
        if self.evaluation_mode not in {"scalar", "batch_native"}:
            raise ValueError("evaluation_mode must be scalar or batch_native")
        if self.evaluation_mode == "batch_native" and not self.batch_sizes:
            raise ValueError("batch_native mode requires batch_sizes")
        if any(int(size) <= 0 for size in self.batch_sizes):
            raise ValueError("batch_sizes must be positive")


class CPUValueScorePool:
    """Persistent spawn pool returning ordered scalar target values and scores."""

    def __init__(self, config: CPUValueScorePoolConfig) -> None:
        self.config = config
        self._executor: concurrent.futures.ProcessPoolExecutor | None = None
        self._startup_barrier: Any | None = None
        self._startup_metadata: Mapping[str, Any] | None = None
        self._opened = False

    def __enter__(self) -> "CPUValueScorePool":
        if self._opened:
            raise RuntimeError("CPU value/score pool is already open")
        self._opened = True
        return self

    def evaluate(self, rows: Any, *, request_id: str) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
        results, matrix, request, input_hash = self._submit(
            rows,
            request_id=request_id,
            worker_function=(
                _worker_eval_batch
                if self.config.evaluation_mode == "batch_native"
                else _worker_eval
            ),
        )
        values = np.concatenate([np.asarray(result["values"], dtype=np.float64) for result in results])
        scores = np.concatenate([np.asarray(result["scores"], dtype=np.float64) for result in results], axis=0)
        if values.shape != (matrix.shape[0],) or scores.shape != matrix.shape:
            raise RuntimeError("worker value/score shape mismatch")
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(scores)):
            raise FloatingPointError("worker returned nonfinite values or scores")
        return values, scores, self._metadata(
            results, request=request, input_hash=input_hash, mode="value_score"
        )

    def evaluate_values(
        self, rows: Any, *, request_id: str
    ) -> tuple[np.ndarray, Mapping[str, Any]]:
        """Evaluate target values without analytic derivative propagation."""

        results, matrix, request, input_hash = self._submit(
            rows,
            request_id=request_id,
            worker_function=(
                _worker_eval_value_batch
                if self.config.evaluation_mode == "batch_native"
                else _worker_eval_value
            ),
        )
        values = np.concatenate(
            [np.asarray(result["values"], dtype=np.float64) for result in results]
        )
        if values.shape != (matrix.shape[0],) or not np.all(np.isfinite(values)):
            raise FloatingPointError("worker returned invalid target values")
        return values, self._metadata(
            results, request=request, input_hash=input_hash, mode="value_only"
        )

    def _submit(
        self,
        rows: Any,
        *,
        request_id: str,
        worker_function: Any,
    ) -> tuple[list[Mapping[str, Any]], np.ndarray, str, str]:
        if not self._opened:
            raise RuntimeError("CPU value/score pool must be opened as a context manager")
        matrix = np.asarray(rows, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != int(self.config.dimension):
            raise ValueError("rows must have shape [batch, dimension]")
        if matrix.shape[0] == 0 or not np.all(np.isfinite(matrix)):
            raise ValueError("rows must be nonempty and finite")
        request = str(request_id)
        if not request:
            raise ValueError("request_id must be nonempty")
        input_hash = _array_hash(matrix)
        worker_count = min(int(self.config.worker_count), int(matrix.shape[0]))
        if self.config.evaluation_mode == "batch_native":
            if matrix.shape[0] % int(self.config.worker_count) != 0:
                raise ValueError(
                    "batch_native evaluation requires batch size divisible by worker_count"
                )
            worker_count = int(self.config.worker_count)
            shard_size = matrix.shape[0] // worker_count
            if shard_size not in {int(size) for size in self.config.batch_sizes}:
                raise ValueError(
                    f"batch_native worker shard size {shard_size} is not declared"
                )
        # Keep the fail-closed CPU environment in place until the executor has
        # submitted its first tasks.  Spawn inherits the environment at that
        # boundary; restoring it earlier is a GPU-visibility race.
        spawn_environment = _cpu_worker_environment(self.config.cores_per_worker)
        previous_environment = {
            key: os.environ.get(key) for key in spawn_environment
        }
        os.environ.update(spawn_environment)
        try:
            try:
                if self._executor is None:
                    context = multiprocessing.get_context("spawn")
                    self._startup_barrier = context.Barrier(
                        int(self.config.worker_count),
                        timeout=float(self.config.timeout_seconds),
                    )
                    self._executor = concurrent.futures.ProcessPoolExecutor(
                        max_workers=int(self.config.worker_count),
                        mp_context=context,
                        initializer=_worker_init,
                        initargs=(
                            str(self.config.worker_factory_path),
                            dict(self.config.worker_config),
                            int(self.config.cores_per_worker),
                            self._startup_barrier,
                            str(self.config.evaluation_mode),
                            tuple(int(size) for size in self.config.batch_sizes),
                        ),
                    )
                    readiness_futures = [
                        self._executor.submit(_worker_ready)
                        for _ in range(int(self.config.worker_count))
                    ]
                    readiness = [
                        future.result(timeout=float(self.config.timeout_seconds))
                        for future in readiness_futures
                    ]
                    startup_worker_metadata = [
                        dict(result["worker_metadata"]) for result in readiness
                    ]
                    startup_worker_pids = sorted(
                        {int(record["pid"]) for record in startup_worker_metadata}
                    )
                    if len(startup_worker_pids) != int(self.config.worker_count):
                        raise RuntimeError(
                            "CPU value/score pool did not initialize every configured worker"
                        )
                    self._startup_metadata = {
                        "configured_worker_count": int(self.config.worker_count),
                        "startup_worker_pids": startup_worker_pids,
                        "startup_worker_metadata": startup_worker_metadata,
                        "startup_worker_ru_maxrss_sum_bytes": int(
                            sum(int(result["ru_maxrss_bytes"]) for result in readiness)
                        ),
                    }
                futures = []
                for worker_index in range(worker_count):
                    start = (matrix.shape[0] * worker_index) // worker_count
                    stop = (matrix.shape[0] * (worker_index + 1)) // worker_count
                    futures.append(self._executor.submit(worker_function, {
                        "worker_index": worker_index,
                        "item_start": start,
                        "item_stop": stop,
                        "request_id": request,
                        "input_hash": input_hash,
                        "rows": matrix[start:stop],
                    }))
            except BaseException:
                # Readiness failures and interrupts can happen before the
                # result futures exist.  They must still terminate the pool.
                self.abort()
                raise
        finally:
            for key, previous in previous_environment.items():
                if previous is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = previous
        try:
            results = [
                future.result(timeout=float(self.config.timeout_seconds))
                for future in futures
            ]
        except BaseException:
            self.abort()
            raise
        results.sort(key=lambda result: int(result["item_start"]))
        if any(
            result["input_hash"] != input_hash or result["request_id"] != request
            for result in results
        ):
            raise RuntimeError("worker input identity mismatch")
        expected_start = 0
        for result in results:
            start = int(result["item_start"])
            stop = int(result["item_stop"])
            if start != expected_start or stop <= start:
                raise RuntimeError("worker shard coverage mismatch")
            if result["shard_hash"] != _array_hash(matrix[start:stop]):
                raise RuntimeError("worker shard content mismatch")
            expected_start = stop
        if expected_start != int(matrix.shape[0]):
            raise RuntimeError("worker shard coverage is incomplete")
        return results, matrix, request, input_hash

    def _metadata(
        self,
        results: list[Mapping[str, Any]],
        *,
        request: str,
        input_hash: str,
        mode: str,
    ) -> Mapping[str, Any]:
        if self._startup_metadata is None:
            raise RuntimeError("CPU value/score pool startup metadata is unavailable")
        return {
            "request_id": request,
            "input_hash": input_hash,
            "worker_count": len(results),
            "worker_pids": sorted({int(result["worker_metadata"]["pid"]) for result in results}),
            "worker_metadata": [dict(result["worker_metadata"]) for result in results],
            "worker_runtime_seconds": [
                float(result["runtime_seconds"]) for result in results
            ],
            "worker_runtime_max_seconds": float(
                max(float(result["runtime_seconds"]) for result in results)
            ),
            "active_worker_ru_maxrss_sum_bytes": int(
                sum(
                    max(
                        int(row["ru_maxrss_bytes"])
                        for row in results
                        if int(row["worker_metadata"]["pid"]) == pid
                    )
                    for pid in {
                        int(row["worker_metadata"]["pid"]) for row in results
                    }
                )
            ),
            "backend": "persistent_cpu_worker_value_score_custom_gradient_bridge",
            "evaluation_mode": str(mode),
            "configured_batch_sizes": [int(size) for size in self.config.batch_sizes],
            "worker_shard_sizes": [
                int(result["item_stop"]) - int(result["item_start"])
                for result in results
            ],
            "jit_compile": False,
            **dict(self._startup_metadata),
        }

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        self._startup_barrier = None
        self._startup_metadata = None
        self._opened = False

    def abort(self) -> None:
        """Fail closed on interruption without waiting for worker tasks."""

        executor = self._executor
        if executor is None:
            self._opened = False
            return
        processes = list(getattr(executor, "_processes", {}).values())
        executor.shutdown(wait=False, cancel_futures=True)
        for process in processes:
            try:
                if process.is_alive():
                    process.terminate()
            except (OSError, ProcessLookupError):
                pass
        for process in processes:
            try:
                process.join(timeout=5.0)
            except (OSError, ProcessLookupError):
                continue
            # A stuck initializer or native call can ignore SIGTERM.  Do not
            # leave a TensorFlow worker behind after an interrupted run.
            try:
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5.0)
            except (OSError, ProcessLookupError):
                pass
        self._executor = None
        self._startup_barrier = None
        self._startup_metadata = None
        self._opened = False

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc, traceback
        if exc_type is None:
            self.close()
        else:
            self.abort()


__all__ = ["CPUValueScorePool", "CPUValueScorePoolConfig"]
